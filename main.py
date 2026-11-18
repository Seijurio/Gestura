"""Gestura — effets caméra en temps réel pilotés par les gestes.

Lancer :  ./run.sh            (ou .venv/bin/python main.py)
          ./run.sh --setup    refaire la configuration
          ./run.sh --4k       rendu 4K (~30 i/s) · ./run.sh --1080p pour revenir au 1080p 60 i/s (par défaut)
"""
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import icons
import ui
from audio import Audio
from arcade import GAMES, Arcade
from live import Live
from onboarding import Onboarding, Unlock, _Fader
from ringlight import RingLight
from tracking import AsyncTracker, Camera

ROOT = Path(__file__).resolve().parent
PROFILE = ROOT / "profile.json"
SETTINGS = ROOT / "settings.json"
CAPTURES = ROOT / "captures"
MODELS = ROOT / "models"
WIN = "Gestura"
TARGET_FPS = 60


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


class Splash:
    """Écran d'initialisation : les modèles se chargent en arrière-plan pendant l'animation."""

    STEPS = [("camera", "Caméra"), ("hands", "Détection des mains"), ("face", "Détection du visage"),
             ("pose", "Suivi du corps"), ("seg", "Détourage de la silhouette"), ("audio", "Son et micro"),
             ("warm", "Calibrage des capteurs")]
    needs_face = needs_hands = needs_seg = False

    def __init__(self, text, render, cam_size):
        self.text = text
        self.render = render
        self.cam_size = cam_size
        self.done = {"camera"}
        self.current = "hands"
        self.tracker = None
        self.audio = None
        self.error = None
        self.t0 = time.monotonic()
        self._thread = None
        self._frame = None
        self.ready_t = None

    def _progress(self, step):
        if self.current:
            self.done.add(self.current)
        self.current = step

    def _load(self):
        try:
            tr = AsyncTracker(MODELS, on_progress=self._progress)
            self._progress("audio")
            self.audio = Audio()
            threading.Thread(target=icons.prewarm, daemon=True).start()  # icônes dessinées en arrière-plan
            self._progress("warm")
            tr.warmup(self._frame)
            self.done.add("warm")
            self.current = None
            self.tracker = tr
        except Exception as e:  # noqa: BLE001
            self.error = e

    def key(self, k):
        return "quit" if k in (ord("q"), 27) else None

    def update(self, frame, _tracker, dt):
        if self._thread is None:
            self._frame = frame.copy()
            self._thread = threading.Thread(target=self._load, daemon=True)
            self._thread.start()
        h, w = frame.shape[:2]
        T = self.text
        t = time.monotonic() - self.t0
        frame[:] = cv2.convertScaleAbs(ui.soft_blur(frame, 22 * ui.S), alpha=0.3)
        c = (w // 2, int(h * 0.36))
        if self.tracker is None:
            ui.spinner(frame, c, ui.u(38), t)
        else:
            if self.ready_t is None:
                self.ready_t = time.monotonic()
            e = time.monotonic() - self.ready_t
            ui.arc(frame, c, ui.u(38), ui.ease_out(e / 0.4), ui.GREEN, 5)
            ui.checkmark(frame, c, ui.u(44), (e - 0.15) / 0.4, ui.GREEN, 6)
        T.draw(frame, "Gestura", (w // 2, int(h * 0.36) + ui.u(70)), 44, ui.WHITE, "mt", bold=True)
        label = "4K" if self.render[1] >= 2160 else f"{self.render[1]}p"
        T.draw(frame, f"Initialisation · rendu {label} · caméra {self.cam_size[0]}x{self.cam_size[1]}",
               (w // 2, int(h * 0.36) + ui.u(124)),
               20, ui.WHITE, "mt", alpha=0.6)
        y = int(h * 0.36) + ui.u(175)
        x = w // 2 - ui.u(150)
        for i, (key, label) in enumerate(self.STEPS):
            a = ui.ease_out((t - i * 0.12) / 0.4)
            yy = y + i * ui.u(34)
            if key in self.done:
                ui.checkmark(frame, (x, yy + ui.u(11)), ui.u(18), 1.0, ui.GREEN, 3)
                col, al = ui.WHITE, 0.95
            elif key == self.current:
                ui.spinner(frame, (x, yy + ui.u(11)), ui.u(8), t, ui.WHITE)
                col, al = ui.WHITE, 0.95
            else:
                cv2.circle(frame, (x, yy + ui.u(11)), ui.u(4), ui.GREY, -1, cv2.LINE_AA)
                col, al = ui.WHITE, 0.4
            T.draw(frame, label, (x + ui.u(24), yy), 20, col, "lt", alpha=al * a)
        if self.error is not None:
            T.draw(frame, f"Erreur : {self.error}", (w // 2, h - ui.u(40)), 18, ui.RED, "mm")
        if self.tracker is not None and time.monotonic() - self.ready_t > 0.8 and t > 1.6:
            return "done"
        return None


def sharpen(frame, amount=0.3):
    """Affûtage léger (masque flou) : rend l'image de la webcam plus nette sans bruit visible."""
    blur = cv2.GaussianBlur(frame, (0, 0), 1.2)
    return cv2.addWeighted(frame, 1 + amount, blur, -amount, 0)


def prepare_frame(raw, render, ring=None):
    """Image caméra → image de travail (miroir, exposition auto si sombre, affûtage, format du rendu).
    Une fois par image caméra."""
    frame = cv2.flip(raw, 1)
    if ring is not None:
        ring.measure(frame)
        frame = ring.expose(frame)
    dark = ring is not None and ring.lum < 80
    if (frame.shape[1], frame.shape[0]) == tuple(render):
        return sharpen(frame, 0.12 if dark else 0.3)  # moins d'affûtage quand c'est sombre (sinon bruit)
    return fit_frame(frame, render)  # (l'agrandissement inclut déjà un affûtage)


_disp = {"t": 0.0, "size": None}


def for_display(frame):
    """Prépare l'image EXACTEMENT au format de la zone d'affichage de la fenêtre.

    macOS dessine l'image en partant du bas de la fenêtre : si son format ne colle pas pile à celui de la
    fenêtre (plein écran sur un écran à encoche, fenêtre redimensionnée…), il reste une bande grise en haut.
    On compose donc nous-mêmes une image à la taille réelle de la fenêtre (Retina = 2 pixels par point) :
    l'image est centrée, agrandie en bicubique, avec des marges noires si les formats diffèrent."""
    now = time.monotonic()
    if now - _disp["t"] > 0.5:
        _disp["t"] = now
        try:
            _, _, ww, wh = cv2.getWindowImageRect(WIN)
            _disp["size"] = (ww * 2, wh * 2) if ww > 0 and wh > 0 else None
        except cv2.error:
            _disp["size"] = None
    if _disp["size"] is None:
        return frame
    tw, th = _disp["size"]
    if tw > 3840:  # au-delà, inutile (et coûteux) : macOS agrandira l'image entière, au bon format
        th, tw = int(th * 3840 / tw), 3840
    h, w = frame.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_LINEAR
    img = frame if (nw, nh) == (w, h) else cv2.resize(frame, (nw, nh), interpolation=interp)
    if abs(nw - tw) <= 1 and abs(nh - th) <= 1:
        return img
    canvas = np.zeros((th, tw, 3), np.uint8)
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = img[:th - y0, :tw - x0]
    return canvas


def fit_frame(frame, size):
    """Met l'image de la caméra au format du rendu (recadrage 16:9 si besoin, puis mise à l'échelle)."""
    h, w = frame.shape[:2]
    W, H = size
    if (w, h) == (W, H):
        return frame
    target = W / H
    if abs(w / h - target) > 0.01:
        if w / h > target:
            nw = int(h * target)
            frame = frame[:, (w - nw) // 2:(w - nw) // 2 + nw]
        else:
            nh = int(w / target)
            frame = frame[(h - nh) // 2:(h - nh) // 2 + nh]
    if frame.shape[1] > W:
        return cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    # agrandissement : on renforce d'abord les détails à la résolution caméra (4x moins cher qu'en 4K),
    # puis interpolation cubique (plus nette que la linéaire) — ~6 ms pour 1080p → 4K sur Apple Silicon
    blur = cv2.GaussianBlur(frame, (0, 0), 1.0)
    frame = cv2.addWeighted(frame, 1.4, blur, -0.4, 0)
    return cv2.resize(frame, (W, H), interpolation=cv2.INTER_CUBIC)


def main():
    if ("--reset" in sys.argv or "--setup" in sys.argv) and PROFILE.exists():
        PROFILE.unlink()
    for m in ("hand_landmarker.task", "face_landmarker.task", "selfie_segmenter.tflite", "pose_landmarker_full.task"):
        if not (MODELS / m).exists():
            sys.exit(f"Modèle manquant : models/{m}")

    settings = load_json(SETTINGS) or {}
    if "--1080p" in sys.argv:
        settings["render"] = "1080p"
    elif "--4k" in sys.argv:
        settings["render"] = "4k"
    render = (3840, 2160) if settings.get("render", "1080p") == "4k" else (1920, 1080)
    try:
        # la webcam du Mac filme au mieux en 1920x1080 à 30 i/s (format natif = meilleure qualité) ;
        # lui demander 4K ou 60 i/s la fait basculer sur un format moins bon (ex. carré 1552x1552)
        cam = Camera(size=(1920, 1080), fps=30)
    except RuntimeError:
        sys.exit("Impossible d'ouvrir la caméra.\n→ Réglages Système › Confidentialité et sécurité › Caméra : "
                 "autorise Terminal / VS Code, puis relance.")
    print(f"Caméra : {cam.size[0]}x{cam.size[1]} @ {cam.fps:.0f} fps → rendu {render[0]}x{render[1]}")
    w, h = render
    ui.set_scale(h / 720)
    text = ui.Text()
    profile = load_json(PROFILE)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, int(1280 * h / w))
    fullscreen = False

    scene = Splash(text, render, cam.size)
    tracker = None
    audio = None
    live = None
    arcade = None
    fader = _Fader()
    shown = None
    seq, last = 0, time.monotonic()

    def switch(new):
        nonlocal scene, settings, live
        if isinstance(scene, Live):
            settings = scene.settings()
            save_json(SETTINGS, settings)
            if not isinstance(new, Arcade):
                scene.close()
        if isinstance(new, Live):
            live = new
        fader.start(shown)
        scene = new

    def new_live():
        return Live(text, profile, settings, CAPTURES, audio)

    # Boucle d'affichage à 60 Hz, découplée de la caméra (comme un jeu vidéo) : les animations, effets
    # et l'interface tournent à 60 i/s ; l'image caméra est remplacée dès qu'une nouvelle arrive.
    target_dt = 1.0 / TARGET_FPS
    ring = RingLight(settings.get("ringlight", "auto"))
    base_frame = None
    cam_seen = time.monotonic()
    try:
        while True:
            t_loop = time.monotonic()
            cseq, raw = cam.latest()
            new_cam = cseq != seq
            if new_cam:
                seq = cseq
                cam_seen = t_loop
                base_frame = prepare_frame(raw, render, ring)
            elif base_frame is None or t_loop - cam_seen > 3:
                if t_loop - cam_seen > 3 and base_frame is not None:
                    print("La caméra ne renvoie plus d'images.")
                    break
                cv2.waitKey(5)
                continue
            frame = base_frame.copy()
            now = time.monotonic()
            dt, last = min(now - last, 0.1), now

            if tracker is not None and new_cam:  # détection seulement sur les nouvelles images caméra
                tracker.submit(frame, hands=scene.needs_hands, face=scene.needs_face, seg=scene.needs_seg,
                               pose=getattr(scene, "needs_pose", False))
            ui.CAM_FPS = getattr(cam, "real_fps", 0.0)
            res = scene.update(frame, tracker, dt)

            if isinstance(scene, Splash) and res == "done":
                tracker, audio = scene.tracker, scene.audio
                switch(Onboarding(text) if profile is None else Unlock(text, profile))
            elif isinstance(scene, Onboarding) and res and res[0] == "done":
                profile = res[1]
                save_json(PROFILE, profile)
                print(f"Profil enregistré → {PROFILE}")
                switch(new_live())
            elif isinstance(scene, Unlock) and res == "done":
                switch(new_live())
            elif isinstance(scene, Arcade) and isinstance(res, tuple) and res[0] == "launch":
                switch(GAMES[res[1]][4](text, audio, arcade.records))

            fader.apply(frame, 0.45)
            shown = frame
            ring.update(dt)
            ui.RING_LEVEL = ring.level
            cv2.imshow(WIN, for_display(ring.apply(frame)))  # cadre lumineux : à l'affichage seulement

            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                if k == ord("i"):  # lumière d'appoint : auto → allumée → éteinte
                    msg = ring.cycle()
                    settings["ringlight"] = ring.mode
                    if live is not None:
                        live._base_settings["ringlight"] = ring.mode
                    if hasattr(scene, "toast"):
                        scene.toast(msg)
                    print(msg)
                    continue
                if k == ord("e"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                    continue
                act = scene.key(k)
                if act == "quit":
                    break
                if act == "skip":
                    switch(new_live())
                elif act == "arcade":
                    arcade = Arcade(text, audio, scene.records)
                    switch(arcade)
                elif isinstance(act, tuple) and act[0] == "launch" and arcade is not None:
                    switch(GAMES[act[1]][4](text, audio, arcade.records))
                elif act == "exit" and isinstance(scene, Arcade) and live is not None:
                    switch(live)
                elif act == "exit" and arcade is not None:
                    switch(arcade)
                elif act == "recalibrate" and tracker is not None:
                    switch(Onboarding(text))
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            spare = target_dt - (time.monotonic() - t_loop)
            if spare > 0.001:
                time.sleep(spare)
    finally:
        if live is not None:
            save_json(SETTINGS, live.settings())
            live.close()
        cam.close()
        if audio is not None:
            audio.close()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
