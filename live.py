"""Mode live : cadre aux doigts + effets + gestes + réactions + dessin néon + menu + photo / vidéo."""
import collections
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import icons
import ui
from drawing import AirCanvas
import effects as effects_mod
from effects import COLOR_MODES, EFFECTS
from gestures import EMOJI, GestureEngine
from menu import RadialMenu
from reactions import Reactions
from tracking import Smoother, match_order, order_tlbr, quad_area

EFFECT_NAMES = [e[0] for e in EFFECTS]
SWITCH_COOLDOWN = 0.45  # délai minimum entre deux changements d'effet par geste

HELP = [
    ("Gestes", ""),
    ("2 mains", "pouces + index = le cadre"),
    ("1 main", "rectangle entre pouce et index"),
    ("glisser", "main ouverte, geste ample et rapide vers la gauche / droite : effet suivant / précédent"),
    ("flèches ‹ ›", "approche l'index d'un bord : la flèche apparaît, pointe-la 0,35 s (reste pour défiler)"),
    ("main ouverte", "immobile 1 s : menu circulaire (poing pour choisir)"),
    ("poing 1 s", "figer / libérer le cadre"),
    ("salut vulcain · clap clap", "photo avec compte à rebours (aperçu polaroïd)"),
    ("pouce levé, rock, cœur", "réactions : confettis, lasers, pluie de cœurs"),
    ("applaudir", "onde de choc (détecté au son du micro + position des mains)"),
    ("", ""),
    ("Touches", ""),
    ("1-9, 0 · N B", "choisir un effet · suivant / précédent"),
    ("[  ]  ·  C", "taille · couleur (ascii, matrix)"),
    ("M", "masque : l'effet épouse ton visage (floutage)"),
    ("D · Z · X", "dessin néon (index tendu dessine, main ouverte gomme) · annuler · effacer"),
    ("L", "verrouiller les gestes de changement d'effet"),
    ("G · F · H · P", "capturer le fond · figer · squelette · étincelles"),
    ("S · V · O", "photo · vidéo · boomerang"),
    ("J", "salle d'arcade : Danse, Bulles, Tranche, Jacques a dit, Esquive"),
    ("E · I · Y", "plein écran · lumière d'appoint (auto / allumée / éteinte) · profondeur auto"),
    ("R · Q", "recalibrer · quitter"),
]

HOLD_LABEL = {"fist": "figer", "thumbs_up": "👍", "rock": "🤘", "vulcan": "🖖", "open": "menu", "heart": "❤️"}


# =========================================================================== utilitaires
class History:
    """Mémoire des ~3 dernières secondes (une image sur deux) pour les effets temporels."""

    def __init__(self, seconds=3.2, rate=15.0):
        self.buf = collections.deque()
        self.seconds, self.period, self.last = seconds, 1.0 / rate, 0.0

    def push(self, frame, mask):
        now = time.monotonic()
        if now - self.last < self.period:  # au temps, pas à l'image (l'affichage tourne à 60 Hz)
            return
        self.last = now
        h, w = frame.shape[:2]
        if w > 1920:  # en 4K on garde la mémoire en 1080p (sinon > 1 Go)
            f = cv2.resize(frame, (1920, int(1920 * h / w)), interpolation=cv2.INTER_AREA)
        else:
            f = frame.copy()
        self.buf.append((now, f, None if mask is None else mask.copy()))
        while self.buf and now - self.buf[0][0] > self.seconds:
            self.buf.popleft()

    def at(self, delay):
        if not self.buf:
            return None, None
        target = time.monotonic() - delay
        for t, f, m in reversed(self.buf):
            if t <= target:
                return f, m
        return self.buf[0][1], self.buf[0][2]

    def clear(self):
        self.buf.clear()


def _writer(path, size, fps=30):
    for cc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError("Impossible de créer la vidéo")


class Recorder:
    """Enregistrement vidéo dans un thread (rythme réel conservé : 30 images/s)."""

    FPS = 30

    def __init__(self, path: Path, size):
        self.w = _writer(path, size, self.FPS)
        self.path = path
        self.t0 = time.monotonic()
        self.count = 0
        self.q = queue.Queue(maxsize=120)
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def write(self, frame):
        due = int((time.monotonic() - self.t0) * self.FPS) + 1
        f = None
        while self.count < due:
            if f is None:
                f = frame.copy()
            try:
                self.q.put_nowait(f)
            except queue.Full:
                pass
            self.count += 1

    @property
    def elapsed(self):
        return time.monotonic() - self.t0

    def _loop(self):
        while True:
            f = self.q.get()
            if f is None:
                break
            self.w.write(f)

    def close(self):
        self.q.put(None)
        self.th.join(timeout=10)
        self.w.release()


def save_async(path: Path, frame):
    threading.Thread(target=cv2.imwrite, args=(str(path), frame.copy(), [cv2.IMWRITE_JPEG_QUALITY, 95]),
                     daemon=True).start()


def save_boomerang(path: Path, frames):
    def run():
        h, w = frames[0].shape[:2]
        wr = _writer(path, (w, h), 30)
        seq = frames + frames[-2:0:-1]
        for _ in range(3):
            for f in seq:
                wr.write(f)
        wr.release()
    threading.Thread(target=run, daemon=True).start()


# =========================================================================== scène
class Live:
    def __init__(self, text, profile, settings, captures: Path, audio=None):
        self.text = text
        self.captures = captures
        name = settings.get("effect", "pixel")
        self.fx = EFFECT_NAMES.index(name) if name in EFFECT_NAMES else 0
        self.cell = int(settings.get("cell", 14))
        self.cell_f = float(self.cell)
        self.color = settings.get("color", "couleur")
        self.locked = bool(settings.get("locked", False))
        thr = float((profile or {}).get("pinch_threshold", 0.35))
        self.audio = audio
        self.engine = GestureEngine(min(thr, 0.42), audio)
        self.records = dict(settings.get("records", {}))
        self._base_settings = dict(settings)
        self.depth_auto = bool(settings.get("profondeur_auto", True))
        self.depth_want = 0.0   # 0..1 : tête trop près → flou d'arrière-plan
        self.depth = 0.0

        now = time.monotonic()
        self.smoother = Smoother(0.35, 0.95, 30)
        self.quad, self.lost = None, 0
        self.frozen = self.show_hands = self.face_mode = self.draw_mode = self.sparkles = self.help = False
        self.last_switch = 0.0
        self.flash_t, self.t_start = now, now
        self.fps = 0.0
        self.state = {}
        self.prev = None
        self.history = History()
        self.bg = None
        self.bg_capture_t = None
        self.countdown_t, self.last_photo = None, 0.0
        self.snap_t = -10.0
        self.rec = None
        self.boom = None
        self.canvas = AirCanvas()
        self.reactions = Reactions()
        self.particles = ui.Particles()
        self.menu = None
        self.toast_msg, self.toast_t = "", -10.0
        self.carousel = float(self.fx)
        self._want_photo = self._want_rec = False
        self._hud_t, self._hud_fps = 0.0, (0, 0)
        self.swipe_fx = (None, -10.0)
        self.polaroid = None
        self._last_tick = 4
        self.edge = {"side": None, "t": 0.0, "last": 0.0}  # boutons ‹ › sur les bords
        self.quad_hist = collections.deque(maxlen=150)  # ~2,5 s de cadres, pour figer "avant le geste"

    # ------------------------------------------------------------------ besoins de détection
    @property
    def _fx_needs(self):
        return EFFECTS[self.fx][2]

    needs_hands = True

    @property
    def needs_face(self):
        # (la profondeur automatique a besoin de savoir si la tête est trop près)
        return self.face_mode or "face" in self._fx_needs or self.depth_auto

    @property
    def needs_pose(self):
        return "pose" in self._fx_needs

    @property
    def needs_seg(self):
        return "seg" in self._fx_needs or (self.depth_auto and self.depth_want > 0)

    def settings(self):
        return {**self._base_settings, "effect": EFFECT_NAMES[self.fx], "cell": self.cell, "color": self.color,
                "locked": self.locked, "records": self.records, "profondeur_auto": self.depth_auto}

    # ------------------------------------------------------------------ actions
    def toast(self, msg):
        self.toast_msg, self.toast_t = msg, time.monotonic()

    def _select(self, i):
        self.fx = i % len(EFFECTS)
        self.flash_t = time.monotonic()
        if "hist" not in self._fx_needs:
            self.history.clear()

    def _gesture_switch(self, step):
        now = time.monotonic()
        if now - self.last_switch < SWITCH_COOLDOWN:
            return
        self.last_switch = now
        self._select(self.fx + step)

    def _set_cell(self, v):
        self.cell_f = float(min(48, max(6, v)))
        self.cell = int(round(self.cell_f / 2) * 2)

    def _freeze_before(self, seconds):
        """Fige le cadre tel qu'il était AVANT le geste (former un poing ou un V casse le cadre)."""
        target = time.monotonic() - seconds
        q = None
        for t, qq in reversed(self.quad_hist):
            q = qq
            if t <= target:
                break
        if q is not None:
            self.quad = q.copy()
            self.smoother.value = q.copy()
        if self.quad is not None:
            self.frozen = True
            return True
        return False

    def _start_countdown(self, auto_freeze=False):
        if self.countdown_t is None and time.monotonic() - self.last_photo > 2.0:
            self.countdown_t = time.monotonic()
            self._last_tick = 4
            # le geste photo casse le cadre : on garde l'effet tel qu'il était, le temps de la photo
            self._unfreeze_after_photo = False
            if auto_freeze and not self.frozen and not self.face_mode and self._freeze_before(1.2):
                self._unfreeze_after_photo = True
            self.toast("Photo dans 3 secondes… (baisse les mains si tu veux)")

    def _sfx(self, data, vol=0.6):
        if self.audio is not None:
            self.audio.play(data, vol)

    def _photo(self, frame):
        from audio import sfx_shutter
        self.captures.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.captures / f"photo_{stamp}.jpg"
        save_async(path, frame)
        self.snap_t = self.last_photo = time.monotonic()
        h, w = frame.shape[:2]
        self.polaroid = (cv2.resize(frame, (int(w * 0.26), int(h * 0.26)), interpolation=cv2.INTER_AREA),
                         self.snap_t)
        self._sfx(sfx_shutter(), 0.8)
        self.toast(f"Photo : captures/{path.name}")
        print(f"Photo → {path}")

    def _draw_polaroid(self, frame):
        """Aperçu de la dernière photo : sort du centre, se pose en bas à droite, puis s'en va."""
        if self.polaroid is None:
            return
        thumb, t0 = self.polaroid
        e = time.monotonic() - t0
        if e > 3.2:
            self.polaroid = None
            return
        h, w = frame.shape[:2]
        th, tw = thumb.shape[:2]
        b = ui.u(12)
        card = np.full((th + b * 2 + ui.u(34), tw + b * 2, 3), 245, np.uint8)
        card[b:b + th, b:b + tw] = thumb
        ui.Text_default().draw(card, time.strftime("%d/%m %H:%M"), (card.shape[1] // 2, th + b + ui.u(8)), 16,
                               (60, 60, 60), "mt", bold=True)
        k = ui.ease_out(min(1.0, (e - 0.25) / 0.55)) if e > 0.25 else 0.0
        out = ui.ease_in_out((e - 2.7) / 0.5) if e > 2.7 else 0.0
        scale = 1.6 - 0.6 * k
        ang = -8 * k
        cx = w / 2 + (w - card.shape[1] * 0.6 - ui.u(30) - w / 2) * k + out * w * 0.4
        cy = h / 2 + (h - card.shape[0] * 0.6 - ui.u(40) - h / 2) * k
        M = cv2.getRotationMatrix2D((card.shape[1] / 2, card.shape[0] / 2), ang, scale * 0.6)
        M[0, 2] += cx - card.shape[1] / 2
        M[1, 2] += cy - card.shape[0] / 2
        warped = cv2.warpAffine(card, M, (w, h))
        mask = cv2.warpAffine(np.full(card.shape[:2], 255, np.uint8), M, (w, h))
        shadow = cv2.GaussianBlur(mask, (0, 0), ui.u(8))
        shadow = np.roll(np.roll(shadow, ui.u(8), 0), ui.u(6), 1)
        cv2.copyTo(cv2.convertScaleAbs(frame, alpha=0.6), shadow, frame)
        cv2.copyTo(warped, mask, frame)

    def _toggle_rec(self, size):
        if self.rec:
            self.rec.close()
            self.toast(f"Vidéo : captures/{self.rec.path.name}")
            print(f"Vidéo → {self.rec.path}")
            self.rec = None
            return
        self.captures.mkdir(exist_ok=True)
        try:
            self.rec = Recorder(self.captures / time.strftime("video_%Y%m%d_%H%M%S.mp4"), size)
            self.toast("Enregistrement…")
        except RuntimeError as e:
            self.toast(str(e))

    def close(self):
        if self.rec:
            self.rec.close()
            print(f"Vidéo → {self.rec.path}")
            self.rec = None

    def key(self, k):
        if k in (ord("q"), 27):
            return "quit"
        if ord("1") <= k <= ord("9"):
            self._select(k - ord("1"))
        elif k == ord("0"):
            self._select(9)
        elif k == ord("n"):
            self._select(self.fx + 1)
        elif k == ord("b"):
            self._select(self.fx - 1)
        elif k == ord("]"):
            self._set_cell(self.cell + 2)
        elif k == ord("["):
            self._set_cell(self.cell - 2)
        elif k == ord("f"):
            self.frozen = not self.frozen
        elif k == ord("h"):
            self.show_hands = not self.show_hands
        elif k == ord("c"):
            i = COLOR_MODES.index(self.color) if self.color in COLOR_MODES else -1
            self.color = COLOR_MODES[(i + 1) % len(COLOR_MODES)]
            self.toast(f"Couleur : {self.color}")
        elif k == ord("m"):
            self.face_mode = not self.face_mode
            self.smoother.reset()
            self.toast("Masque visage : " + ("ON" if self.face_mode else "OFF"))
        elif k == ord("d"):
            self.draw_mode = not self.draw_mode
            self.toast("Dessin : index tendu pour dessiner, main ouverte pour gommer" if self.draw_mode
                       else "Dessin : OFF (X pour effacer)")
        elif k == ord("x"):
            self.canvas.clear()
        elif k == ord("z"):
            self.canvas.undo()
        elif k == ord("l"):
            self.locked = not self.locked
            self.toast("Gestes de changement " + ("verrouillés" if self.locked else "déverrouillés"))
        elif k == ord("p"):
            self.sparkles = not self.sparkles
        elif k == ord("y"):
            self.depth_auto = not self.depth_auto
            self.toast("Profondeur automatique : " + ("ON (tête trop près = fond flouté)" if self.depth_auto else "OFF"))
        elif k == ord("g"):
            self.bg_capture_t = time.monotonic()
        elif k == ord("o"):
            if self.boom is None:
                self.boom = {"t0": time.monotonic(), "frames": []}
        elif k == 9:  # Tab
            self.help = not self.help
        elif k == ord("r"):
            return "recalibrate"
        elif k in (ord("j"), ord("k")):
            return "arcade"
        elif k == ord("s"):
            self._want_photo = True
        elif k == ord("v"):
            self._want_rec = True
        return None

    # ------------------------------------------------------------------ gestes
    def _handle_events(self, events, hands, w, h):
        for ev in events:
            kind = ev[0]
            if kind == "swipe":
                self._gesture_switch(+1 if ev[1] == "left" else -1)
                self.swipe_fx = (ev[1], time.monotonic())
            elif kind == "pinch_tap":
                self._gesture_switch(+1)
            elif kind == "hold":
                pose, hd = ev[1], ev[2]
                pos = hd.pts[9]
                if pose == "fist":
                    if self.frozen:
                        self.frozen = False
                        self.toast("Cadre libéré")
                    else:
                        self._freeze_before(1.3)
                        self.toast("Cadre figé (refais un poing pour libérer)")
                elif pose == "thumbs_up":
                    self.reactions.thumbs(pos)
                elif pose == "rock":
                    self.reactions.rock(pos)
                elif pose == "vulcan":  # 🖖 = photo
                    self._start_countdown(auto_freeze=True)
                elif (pose == "open" and not self.draw_mode and len(hands) == 1 and self.menu is None
                      and "full" not in self._fx_needs):  # (avec l'armure, main ouverte = propulseur)
                    self.menu = RadialMenu(EFFECTS, self.fx, pos, (w, h))
            elif kind == "heart":
                self.reactions.hearts(ev[1])
            elif kind == "clap":
                self.reactions.clap(ev[1] if ev[1] is not None else (w / 2, h / 2))
            elif kind == "double_clap":
                self._start_countdown(auto_freeze=True)

    def _edge_geom(self, w, h):
        r = ui.u(54)
        return r, ((ui.u(78), h // 2), (w - ui.u(78), h // 2))

    def _edge_buttons(self, hands, w, h, now):
        """Boutons ‹ › sur les bords : pointe l'index dessus 0,35 s → effet précédent / suivant
        (reste dessus pour faire défiler). Plus fiable qu'un glissé quand on hésite."""
        e = self.edge
        if self.locked or self.menu is not None or self.draw_mode or self.help or not hands:
            e["side"] = None
            return
        r, (lc, rc) = self._edge_geom(w, h)
        hit = None
        for hd in hands:
            p = hd.pts[8]
            for side, c in (("prev", lc), ("next", rc)):
                if (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 < (r * 1.25) ** 2:
                    hit = side
        if hit != e["side"]:
            e["side"], e["t"] = hit, now
            return
        if hit and now - e["t"] >= 0.35 and now - e["last"] >= 0.75:
            e["last"] = now
            self._select(self.fx + (1 if hit == "next" else -1))
            self.swipe_fx = ("left" if hit == "next" else "right", now)

    def _draw_edge_buttons(self, frame, hands):
        """Flèches invisibles par défaut : elles apparaissent en fondu seulement quand un index s'approche
        du bord, et disparaissent dès qu'il s'éloigne (même pendant le cadrage)."""
        h, w = frame.shape[:2]
        r, centers = self._edge_geom(w, h)
        now = time.monotonic()
        dt = min(0.1, now - self.edge.get("draw_t", now))
        self.edge["draw_t"] = now
        enabled = not (self.locked or self.menu is not None or self.draw_mode or self.help)
        for side, c in zip(("prev", "next"), centers):
            near = enabled and any((hd.pts[8][0] - c[0]) ** 2 + (hd.pts[8][1] - c[1]) ** 2 < (r * 3.0) ** 2
                                   for hd in hands)
            key = "vis_" + side
            self.edge[key] = ui.approach(self.edge.get(key, 0.0), 1.0 if near else 0.0, dt, 12)
            a = self.edge[key]
            if a < 0.03:
                continue
            hov = self.edge["side"] == side
            x0, y0 = int(c[0] - r - ui.u(8)), int(c[1] - r - ui.u(8))
            x1, y1 = int(c[0] + r + ui.u(8)), int(c[1] + r + ui.u(8))
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            roi = frame[y0:y1, x0:x1]
            over = roi.copy()
            ui.panel(over, c[0] - r - x0, c[1] - r - y0, c[0] + r - x0, c[1] + r - y0, 0.6 if hov else 0.4, r)
            s = r * 0.32
            d = -1 if side == "prev" else 1
            cc = (c[0] - x0, c[1] - y0)
            pts = np.array([[cc[0] - d * s * 0.5, cc[1] - s], [cc[0] + d * s * 0.6, cc[1]],
                            [cc[0] - d * s * 0.5, cc[1] + s]], np.int32)
            cv2.polylines(over, [pts], False, ui.WHITE, max(2, ui.u(4)), cv2.LINE_AA)
            if hov:
                ui.arc(over, cc, r + ui.u(5), min(1.0, (now - self.edge["t"]) / 0.35), ui.HUD_GREEN, 4)
            cv2.addWeighted(over, a, roi, 1 - a, 0, roi)

    def _raw_quad(self, hands, face):
        if self.face_mode:
            return face.box() if face is not None else None
        if self.draw_mode:
            return None
        if len(hands) >= 2:
            a, b = hands[0], hands[1]
            return np.array([a.pts[4], a.pts[8], b.pts[4], b.pts[8]], np.float32)
        if len(hands) == 1:
            hd = hands[0]
            if hd.pinch_ratio < self.engine.pinch_thr * 1.4:
                return None
            (x0, y0), (x1, y1) = hd.pts[4], hd.pts[8]
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        return None

    # ------------------------------------------------------------------ boucle
    def update(self, frame, tracker, dt):
        h, w = frame.shape[:2]
        now = time.monotonic()
        T = self.text
        if dt > 0:
            self.fps = 0.92 * self.fps + 0.08 / dt if self.fps else 1 / dt
        hands = tracker.hands()
        face = tracker.face() if self.needs_face else None
        needs = self._fx_needs

        # ---- silhouette, mémoire, fond
        mask_small = tracker.mask() if self.needs_seg else None
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR) if mask_small is not None else None
        # ---- profondeur automatique : tête trop près de la caméra → arrière-plan flouté (bokeh)
        if self.depth_auto and face is not None:
            ratio = face.width / w  # ~0,15-0,2 à bonne distance, > 0,3 quand on est collé à l'écran
            self.depth_want = float(np.clip((ratio - 0.27) / 0.08, 0, 1))
        elif not self.depth_auto or face is None:
            self.depth_want = 0.0
        self.depth = ui.approach(self.depth, self.depth_want, dt, 4)
        if self.depth > 0.03 and mask is not None and EFFECT_NAMES[self.fx] not in ("portrait", "studio"):
            frame[:] = effects_mod.depth_blur(frame, mask, self.depth)

        if "hist" in needs:
            self.history.push(frame, mask_small)
        if "bg" in needs or self.bg_capture_t is not None:
            if self.bg is None or self.bg.shape != frame.shape:
                self.bg = frame.copy()
            if mask_small is not None:
                learn = cv2.dilate((mask_small > 0.15).astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
                learn = cv2.resize(learn.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.copyTo(cv2.addWeighted(self.bg, 0.88, frame, 0.12, 0), learn, self.bg)

        # ---- gestes
        allow = {"hold", "heart", "clap"}  # poing (figer) et V (photo) marchent aussi pendant le cadrage
        # le glissé reste actif même en cadrant : il exige une main ouverte (≥ 3 doigts tendus),
        # alors qu'une main qui fait le cadre (pouce + index) ne l'est pas
        if not self.locked and self.menu is None and not self.draw_mode:
            allow |= {"swipe"}
        if self.menu is not None:
            allow = set()
        events = self.engine.update(hands, tracker.hands_seq, now, (w, h), allow)
        self._handle_events(events, hands, w, h)
        poses = self.engine.poses()
        self._edge_buttons(hands, w, h, now)

        # ---- menu circulaire
        if self.menu is not None:
            main = max(((p, hd) for p, hd in poses.values() if hd is not None), key=lambda x: x[1].palm,
                       default=(None, None))
            res = self.menu.update(main[1], main[0], dt)
            if res is not None:
                if res[0] == "select":
                    self._select(res[1])
                self.menu = None

        # ---- cadre
        if not self.frozen and self.menu is None:
            raw = self._raw_quad(hands, face)
            if raw is not None and quad_area(order_tlbr(raw)) > 1500 * ui.S * ui.S:
                raw = match_order(self.smoother.value, raw)
                self.quad = self.smoother.update(raw)
                self.lost = 0
                self.quad_hist.append((now, self.quad.copy()))
            else:
                self.lost += 1
                if self.lost > 6:
                    self.quad = None
                    self.smoother.reset()

        name, fn, _ = EFFECTS[self.fx]
        outline = hint = None
        full = "full" in needs  # effet plein écran (ex. armure) : pas besoin de cadre
        if full and self.menu is None:
            ctx = {"cell": self.cell, "t": now - self.t_start, "dt": dt, "color": self.color, "quad": None,
                   "center": face.center if face is not None else None, "prev": self.prev, "history": self.history,
                   "mask": mask, "bg": self.bg, "state": self.state, "face": face,
                   "audio": self.audio.level if self.audio is not None else 0.0,
                   "hands": hands, "hand_poses": [v for v in poses.values() if v[1] is not None],
                   "body": tracker.body() if self.needs_pose else None, "audio_obj": self.audio}
            frame[:] = fn(frame, ctx)
        elif self.quad is not None and self.menu is None:
            q = order_tlbr(self.quad)
            ctx = {"cell": self.cell, "t": now - self.t_start, "dt": dt, "color": self.color, "quad": q,
                   "center": q.mean(axis=0), "prev": self.prev, "history": self.history, "mask": mask,
                   "bg": self.bg, "state": self.state, "face": face,
                   "audio": self.audio.level if self.audio is not None else 0.0}
            out = fn(frame, ctx)
            if self.face_mode and face is not None:
                # masque qui épouse le visage, bords adoucis (floutage propre)
                pts = face.pts[:, :2]
                c = pts.mean(axis=0)
                hull = cv2.convexHull(((pts - c) * 1.12 + c).astype(np.float32))
                m = ui.feather_mask(h, w, lambda mk, k: cv2.fillConvexPoly(mk, (hull * k).astype(np.int32), 255,
                                                                           cv2.LINE_AA), ui.u(10))
                frame[:] = cv2.blendLinear(frame, out, 1.0 - m, m)
            elif not self.face_mode:
                mk = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(mk, q.astype(np.int32), 255, cv2.LINE_AA)
                cv2.copyTo(out, mk, frame)
                outline = q  # le contour du cadre est de l'interface : dessiné APRÈS la capture photo
        elif not self.help and self.menu is None and not self.draw_mode and not self.reactions.busy:
            hint = "Montre ton visage" if self.face_mode else "Forme un cadre avec tes pouces et index"

        self._hands_vis = hands

        # ---- dessin
        if self.draw_mode:
            m = self.canvas.update(poses, w, h)
            if m:
                self.toast(m)
        self.canvas.render(frame, False)  # traits seulement (la barre d'outils est dessinée plus bas)

        # ---- étincelles, squelette, réactions
        if self.sparkles:
            for hd in hands:
                self.particles.emit(hd.pts[8], 2, speed=120)
        self.particles.update(dt)
        self.particles.draw(frame)
        self.reactions.update_draw(frame, dt)

        # ===== image "propre" pour les photos / vidéos : effet + dessin + réactions, SANS interface
        clean = frame.copy()

        # ===== à partir d'ici : interface (contour du cadre, squelette, jauges, menus…)
        if outline is not None:
            ui.draw_quad(frame, outline, ui.HUD_GREEN if self.frozen else ui.WHITE)
        self.prev = frame.copy() if name == "tunnel infini" else None
        if hint:
            T.draw(frame, hint, (w // 2, h - ui.u(40)), 24, ui.WHITE, "mm", stroke=2, alpha=0.8)
        if self.draw_mode:
            self.canvas._draw_ui(frame)
        if self.show_hands:
            for key, (pose, hd) in poses.items():
                if hd is not None:
                    ui.draw_hand(frame, hd, alpha=0.7)
                    if pose in EMOJI:
                        ui.blit(frame, icons.get(EMOJI[pose], ui.u(44)), hd.pts[0] + (0, ui.u(40)), 0.9)

        # ---- retour visuel des glissés : traînée pendant le geste, flèche quand il est validé
        for start, cur, prog in self.engine.swipe_trail:
            col = ui.HUD_GREEN if prog >= 1 else ui.WHITE
            over = frame.copy()
            cv2.line(over, (int(start[0]), int(start[1])), (int(cur[0]), int(cur[1])), col, ui.u(10), cv2.LINE_AA)
            cv2.addWeighted(over, 0.25 + 0.35 * prog, frame, 0.75 - 0.35 * prog, 0, frame)
        sd, st = self.swipe_fx
        se = now - st
        if se < 0.5:
            x = w // 2 + (-1 if sd == "left" else 1) * int(ui.ease_out(se / 0.5) * w * 0.25)
            T.draw(frame, "<<" if sd == "left" else ">>", (x, h // 2), 90, ui.WHITE, "mm", bold=True, stroke=3,
                   alpha=1 - se / 0.5)

        # ---- jauges de maintien
        for pos, frac, pose in self.engine.progress:
            if (pose == "open" and (self.draw_mode or len(hands) != 1 or "full" in self._fx_needs)) \
                    or self.menu is not None:
                continue
            if frac > 0.15:
                ui.arc(frame, pos, ui.u(54), frac, ui.WHITE, 5)
                lab = HOLD_LABEL.get(pose, "")
                if lab and not lab.isascii():
                    ui.blit(frame, icons.get(lab, ui.u(40)), pos, 0.9)
                elif lab:
                    T.draw(frame, lab, pos, 18, ui.WHITE, "mm", bold=True, stroke=2)
        if self.menu is not None:
            self.menu.draw(frame, self.cell, self.color, now - self.t_start)

        # ---- compte à rebours photo / capture du fond
        if self.countdown_t is not None:
            left = 3 - (now - self.countdown_t)
            if left <= 0:
                self.countdown_t = None
                self._photo(clean)
                if getattr(self, "_unfreeze_after_photo", False):
                    self.frozen = False
                    self._unfreeze_after_photo = False
            else:
                k = left - int(left)
                if int(left) + 1 != self._last_tick:  # bip à chaque seconde
                    self._last_tick = int(left) + 1
                    from audio import sfx_pop
                    self._sfx(sfx_pop(1.4), 0.4)
                ui.arc(frame, (w // 2, h // 2), ui.u(110), 1 - k, ui.WHITE, 6)
                T.draw(frame, str(int(left) + 1), (w // 2, h // 2), 120 + 60 * k, ui.WHITE, "mm", bold=True,
                       stroke=4, alpha=0.4 + 0.6 * k)
        if self.bg_capture_t is not None:
            left = 3 - (now - self.bg_capture_t)
            if left <= 0:
                self.bg = frame.copy()
                self.bg_capture_t = None
                self.toast("Fond capturé")
            else:
                ui.dim(frame, 0.6)
                T.draw(frame, "Sors du champ !", (w // 2, h // 2 - ui.u(90)), 40, ui.WHITE, "mm", bold=True)
                T.draw(frame, str(int(left) + 1), (w // 2, h // 2 + ui.u(10)), 110, ui.WHITE, "mm", bold=True)

        # ---- photo / vidéo / boomerang (image "propre", sans interface)
        if self._want_photo:
            self._want_photo = False
            self._photo(clean)
        if self._want_rec:
            self._want_rec = False
            self._toggle_rec((w, h))
        if self.rec:
            self.rec.write(clean)
        if self.boom is not None:
            self.boom["frames"].append(cv2.resize(clean, (1280, int(1280 * h / w)), interpolation=cv2.INTER_AREA))
            if now - self.boom["t0"] > 1.6:
                self.captures.mkdir(exist_ok=True)
                path = self.captures / time.strftime("boomerang_%Y%m%d_%H%M%S.mp4")
                save_boomerang(path, self.boom["frames"])
                self.toast(f"Boomerang : captures/{path.name}")
                self.boom = None

        self._draw_polaroid(frame)
        self._hud(frame, name, tracker, dt)
        return None

    # ------------------------------------------------------------------ interface
    def _hud(self, frame, name, tracker, dt):
        self._draw_edge_buttons(frame, getattr(self, "_hands_vis", []))  # interface : pas dans les photos
        h, w = frame.shape[:2]
        T = self.text
        now = time.monotonic()
        x = ui.u(14)
        if now - self._hud_t > 0.5:  # rafraîchi 2x/s (évite de re-rendre le texte)
            self._hud_t = now
            self._hud_fps = (round(self.fps), round(tracker.det_fps))
        fps, dfps = self._hud_fps
        T.draw(frame, f"{name} | {fps} fps | caméra {round(ui.CAM_FPS)} fps | détection {dfps} fps | {w}x{h} | TAB aide",
               (x, ui.u(10)), 19,
               ui.HUD_GREEN, "lt", bold=True, stroke=2)
        badges = []
        if name in ("ascii", "matrix"):
            badges.append(f"C couleur : {self.color}")
        if self.frozen:
            badges.append("FIGÉ")
        if self.face_mode:
            badges.append("MASQUE VISAGE")
        if self.draw_mode:
            badges.append("DESSIN")
        if self.locked:
            badges.append("GESTES VERROUILLÉS")
        if ui.RING_LEVEL > 0.1:
            badges.append("LUMIÈRE D'APPOINT (I)")
        if self.depth > 0.1:
            badges.append("PROFONDEUR AUTO (Y)")
        if self.sparkles:
            badges.append("ÉTINCELLES")
        if self.boom is not None:
            badges.append("BOOMERANG…")
        if name == "invisibilité" and self.bg_capture_t is None:
            badges.append("G = capturer le fond (sors du champ)")
        if badges:
            T.draw(frame, "  ·  ".join(badges), (x, ui.u(38)), 17, ui.WHITE, "lt", bold=True, stroke=2, alpha=0.9)

        if self.rec:
            if int(now * 2) % 2 == 0:
                cv2.circle(frame, (w - ui.u(120), ui.u(24)), ui.u(8), ui.RED, -1, cv2.LINE_AA)
            el = int(self.rec.elapsed)
            T.draw(frame, f"REC {el // 60:02d}:{el % 60:02d}", (w - ui.u(14), ui.u(12)), 20, ui.WHITE, "rt",
                   bold=True, stroke=2)

        # carrousel des effets (glisse avec l'animation)
        self.carousel = ui.approach(self.carousel, self.fx, dt, 12)
        fe = now - self.flash_t
        if fe < 1.8 and not self.help and self.menu is None:
            alpha = 1.0 - ui.ease_in_out((fe - 1.2) / 0.6) if fe > 1.2 else 1.0
            n = len(EFFECTS)
            cy = h - ui.u(78)
            base = int(round(self.carousel))
            for d in range(-3, 4):
                i = base + d
                off = i - self.carousel
                a = max(0.0, 1 - abs(off) / 3.2) * alpha
                cur = (i % n) == (self.fx % n)
                T.draw(frame, EFFECTS[i % n][0], (w // 2 + off * ui.u(190), cy), 30 if cur else 20, ui.WHITE, "mm",
                       bold=cur, stroke=3 if cur else 2, alpha=a)
            T.draw(frame, f"{self.fx % n + 1} / {n}", (w // 2, cy + ui.u(30)), 15, ui.WHITE, "mm", stroke=2,
                   alpha=0.7 * alpha)

        te = now - self.toast_t
        if te < 2.2:
            a = min(1.0, te / 0.15) * (1 - ui.ease_in_out((te - 1.6) / 0.6) if te > 1.6 else 1.0)
            tw, th = T.size(self.toast_msg, 20, True)
            y = ui.u(150) if self.draw_mode else ui.u(70)
            ui.panel(frame, w // 2 - tw // 2 - ui.u(16), y - ui.u(8), w // 2 + tw // 2 + ui.u(16),
                     y + th + ui.u(8), 0.55 * a)
            T.draw(frame, self.toast_msg, (w // 2, y), 20, ui.WHITE, "mt", bold=True, alpha=a)

        if self.help:
            pw, ph = ui.u(900), ui.u(40 + 29 * len(HELP))
            x0, y0 = (w - pw) // 2, (h - ph) // 2
            ui.panel(frame, x0, y0, x0 + pw, y0 + ph, 0.75)
            y = y0 + ui.u(22)
            for k, desc in HELP:
                if k and not desc:
                    T.draw(frame, k.upper(), (x0 + ui.u(28), y), 15, ui.HUD_GREEN, "lt", bold=True)
                elif k:
                    T.draw(frame, k, (x0 + ui.u(28), y), 18, ui.WHITE, "lt", bold=True)
                    T.draw(frame, desc, (x0 + ui.u(210), y), 18, ui.WHITE, "lt", alpha=0.85)
                y += ui.u(29)

        se = now - self.snap_t
        if se < 0.3:
            k = (0.3 - se) / 0.3
            cv2.addWeighted(frame, 1 - k * 0.8, np.full_like(frame, 255), k * 0.8, 0, frame)
