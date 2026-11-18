"""Premier lancement : enregistrement du visage + calibration des mains (style Face ID).
Lancements suivants : scan de "déverrouillage".

Fluidité : fondu enchaîné entre les étapes, valeurs animées avec une interpolation indépendante
du framerate, et seul le détecteur utile à l'étape tourne (visage OU mains).
"""
import math
import time

import cv2
import numpy as np

import ui
from effects import pixel
from tracking import order_tlbr, quad_area, signature_distance

N_TICKS = 72
FACE_MATCH_THRESHOLD = 0.06   # plus petit = plus strict (le score s'affiche pendant le scan)
TURN_SCALE = (0.28, 0.17)     # amplitude de rotation (yaw, pitch) pour remplir une graduation
SPACE, ESC = 32, 27


def _circle_geom(frame):
    h, w = frame.shape[:2]
    return (w // 2, int(h * 0.54)), int(h * 0.27)


def _face_hint(face, center, r):
    if face is None:
        return False, "Place ton visage dans le cercle"
    c = face.center
    if np.hypot(c[0] - center[0], c[1] - center[1]) > 0.35 * r:
        return False, "Centre ton visage"
    if face.width < 0.55 * r:
        return False, "Rapproche-toi un peu"
    if face.width > 1.35 * r:
        return False, "Recule un peu"
    return True, ""


def _face_mesh(frame, face, center, r, t, color=(255, 255, 255)):
    """Nuage de points du visage façon scan 3D (seulement à l'intérieur du cercle)."""
    if face is None:
        return
    pts = face.pts[::5, :2]
    over = frame.copy()
    for i, (x, y) in enumerate(pts):
        if (x - center[0]) ** 2 + (y - center[1]) ** 2 < r * r:
            a = 0.5 + 0.5 * math.sin(t * 4 + i * 0.37)
            cv2.circle(over, (int(x), int(y)), max(1, ui.u(1.6)), color, -1, cv2.LINE_AA)
            if a > 0.85:
                cv2.circle(over, (int(x), int(y)), ui.u(3), ui.GREEN, -1, cv2.LINE_AA)
    cv2.addWeighted(over, 0.55, frame, 0.45, 0, frame)


class _Fader:
    """Fondu enchaîné entre deux étapes."""

    def __init__(self):
        self.img, self.t = None, 0.0

    def start(self, last):
        self.img = None if last is None else last
        self.t = time.monotonic()

    def apply(self, frame, dur=0.4):
        if self.img is None or self.img.shape != frame.shape:
            return
        e = (time.monotonic() - self.t) / dur
        if e >= 1:
            self.img = None
            return
        k = ui.ease_in_out(e)
        cv2.addWeighted(self.img, 1 - k, frame, k, 0, frame)


class _StableText:
    """Évite que les consignes clignotent quand la détection hésite."""

    def __init__(self):
        self.cur, self.cand, self.t = "", "", 0.0

    def __call__(self, msg, delay=0.3):
        now = time.monotonic()
        if msg != self.cand:
            self.cand, self.t = msg, now
        if msg != self.cur and (now - self.t > delay or not self.cur):
            self.cur = msg
        return self.cur


def _titles(T, frame, title, sub, t_phase):
    w = frame.shape[1]
    e = ui.ease_out((time.monotonic() - t_phase) / 0.45)
    off = int((1 - e) * ui.u(14))
    if title:
        T.draw(frame, title, (w // 2, ui.u(34) + off), 40, ui.WHITE, "mt", bold=True, alpha=e)
    if sub:
        T.draw(frame, sub, (w // 2, ui.u(92) + off), 24, ui.WHITE, "mt", alpha=0.9 * e)


# =========================================================================== configuration
class Onboarding:
    def __init__(self, text: ui.Text):
        self.text = text
        self.levels = np.zeros(N_TICKS, np.float32)
        self.base = None
        self.sigs, self.yp = [], []
        self.open_samples, self.pinch_samples, self.palms = [], [], []
        self.profile = {}
        self.fader = _Fader()
        self.last = None
        self.msg = _StableText()
        self.ptr = np.zeros(2, np.float32)
        self.ptr_a = 0.0
        self.disp = 0.0
        self.particles = ui.Particles()
        self.phase = None
        self._set("intro")

    def _set(self, phase):
        if self.phase is not None:
            self.fader.start(self.last)
        self.phase = phase
        self.t_phase = time.monotonic()
        self.msg = _StableText()
        self.hold = 0.0
        self.disp = 0.0
        self._burst = False

    @property
    def needs_face(self):
        return self.phase in ("intro", "face_center", "face_turn")

    @property
    def needs_hands(self):
        return self.phase in ("hand_open", "hand_pinch", "frame")

    needs_seg = False

    def _elapsed(self):
        return time.monotonic() - self.t_phase

    def _step_index(self):
        return {"intro": 0, "face_center": 0, "face_turn": 0, "face_ok": 1, "hand_open": 1,
                "hand_pinch": 2, "frame": 3, "done": 4}[self.phase]

    def key(self, k):
        if k in (ord("q"), ESC):
            return "quit"
        if k == SPACE:
            nxt = {"intro": "face_center", "face_center": "face_ok", "face_turn": "face_ok",
                   "face_ok": "hand_open", "hand_open": "hand_pinch", "hand_pinch": "frame", "frame": "done"}
            if self.phase in nxt:
                self._set(nxt[self.phase])
        return None

    # ------------------------------------------------------------------ boucle
    def update(self, frame, tracker, dt):
        h, w = frame.shape[:2]
        center, r = _circle_geom(frame)
        T = self.text
        p = self.phase
        e = self._elapsed()
        face = tracker.face() if self.needs_face else None
        hands = tracker.hands() if self.needs_hands else []
        r_in, r_out = r + ui.u(18), r + ui.u(32)

        if p in ("intro", "face_center", "face_turn", "face_ok"):
            ui.focus_circle(frame, center, r, dim=0.22)
            cv2.circle(frame, center, r, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            ui.dim(frame, 0.55)

        title = sub = hint = ""

        if p == "intro":
            title = "Configuration"
            sub = "On enregistre ton visage, puis on calibre tes mains"
            hint = "ESPACE pour commencer"
            # les graduations apparaissent une à une
            lv = np.clip(e * 1.6 - np.arange(N_TICKS) / N_TICKS, 0, 1) * 0.35
            ui.ticks_ring(frame, center, r_in, r_out, lv, color_on=ui.WHITE)
            _face_mesh(frame, face, center, r, e)
            if e > 3.0:
                self._set("face_center")

        elif p == "face_center":
            title = "Visage"
            ok, m = _face_hint(face, center, r)
            if ok:
                self.hold += dt
                self.yp.append((face.yaw, face.pitch))
                self.sigs.append(face.signature)
                m = "Ne bouge plus…"
            else:
                self.hold = max(0.0, self.hold - 1.5 * dt)
            sub = self.msg(m)
            self.disp = ui.approach(self.disp, self.hold / 1.2, dt, 14)
            ui.ticks_ring(frame, center, r_in, r_out, np.full(N_TICKS, 0.35), color_on=ui.WHITE)
            ui.arc(frame, center, r + ui.u(4), self.disp, ui.GREEN, 4)
            _face_mesh(frame, face, center, r, e)
            if self.hold >= 1.2:
                self.base = np.median(np.array(self.yp[-40:]), axis=0)
                self._set("face_turn")

        elif p == "face_turn":
            title = "Visage"
            pointer = None
            if face is not None:
                d = np.array([(face.yaw - self.base[0]) / TURN_SCALE[0],
                              (face.pitch - self.base[1]) / TURN_SCALE[1]], np.float32)
                self.ptr = ui.approach(self.ptr, d, dt, 16)
                mag = float(np.hypot(*self.ptr))
                self.ptr_a = ui.approach(self.ptr_a, 1.0 if mag > 0.55 else 0.0, dt, 10)
                if mag > 0.55:
                    pointer = math.atan2(self.ptr[1], self.ptr[0])
                    j0 = int(round(pointer / (2 * math.pi) * N_TICKS))
                    for k in range(-4, 5):
                        j = (j0 + k) % N_TICKS
                        wgt = 1 - abs(k) / 5
                        self.levels[j] = min(1.0, self.levels[j] + dt * 5.0 * wgt * min(1.0, mag))
                elif mag < 0.25:
                    self.sigs.append(face.signature)
                sub = self.msg("Bouge lentement la tête pour compléter le cercle")
            else:
                sub = self.msg("Place ton visage dans le cercle")
            ui.ticks_ring(frame, center, r_in, r_out, 0.2 + 0.8 * self.levels,
                          highlight=pointer if self.ptr_a > 0.3 else None,
                          color_off=(150, 150, 150))
            frac = float((self.levels >= 0.99).mean())
            self.disp = ui.approach(self.disp, frac, dt, 8)
            T.draw(frame, f"{int(round(self.disp * 100))} %", (w // 2, ui.u(136)), 22, ui.GREEN, "mt", bold=True)
            _face_mesh(frame, face, center, r, e, color=(200, 255, 200))
            if frac >= 0.9:
                self.levels[:] = 1.0
                self._set("face_ok")

        elif p == "face_ok":
            title = "Visage enregistré"
            ui.ticks_ring(frame, center, r_in, r_out, np.ones(N_TICKS))
            ui.dim(frame, 1.0 - 0.5 * ui.ease_out(e / 0.4))
            ui.checkmark(frame, center, r * 0.8, e / 0.6, ui.GREEN, 10)
            if e > 1.4:
                if self.sigs:
                    self.profile["face_signature"] = np.median(np.array(self.sigs), axis=0).tolist()
                if self.base is not None:
                    self.profile["face_baseline"] = [float(v) for v in self.base]
                self._set("hand_open")

        elif p == "hand_open":
            title = "Mains"
            hand = max(hands, key=lambda hd: hd.palm) if hands else None
            m = "Montre une main ouverte, doigts écartés"
            if hand is not None:
                ui.draw_hand(frame, hand)
                if hand.is_open:
                    self.hold += dt
                    self.open_samples.append(hand.pinch_ratio)
                    self.palms.append(hand.palm)
                else:
                    self.hold = max(0.0, self.hold - dt)
                    m = "Écarte bien les doigts"
                self.disp = ui.approach(self.disp, self.hold / 1.5, dt, 14)
                ui.arc(frame, hand.pts[9], hand.palm * 1.25, self.disp, ui.GREEN, 6)
            sub = self.msg(m)
            if self.hold >= 1.5:
                self._set("hand_pinch")

        elif p == "hand_pinch":
            title = "Pince"
            sub = "Colle ton pouce et ton index (ça sert à changer d'effet)"
            hand = max(hands, key=lambda hd: hd.palm) if hands else None
            if hand is not None:
                ui.draw_hand(frame, hand)
                if hand.pinch_ratio < 0.45:
                    self.hold += dt
                    self.pinch_samples.append(hand.pinch_ratio)
                    self.particles.emit(hand.pinch_point, 2, hue=0.33, speed=80)
                else:
                    self.hold = max(0.0, self.hold - dt)
                self.disp = ui.approach(self.disp, self.hold / 1.2, dt, 14)
                ui.arc(frame, hand.pinch_point, ui.u(40), self.disp, ui.GREEN, 5)
            if self.hold >= 1.2:
                self._set("frame")

        elif p == "frame":
            title = "Cadre"
            m = "Avec tes deux mains, forme un cadre avec pouces et index"
            for hd in hands:
                ui.draw_hand(frame, hd, alpha=0.45)
            if len(hands) >= 2:
                quad = order_tlbr(np.array([hands[0].pts[4], hands[0].pts[8],
                                            hands[1].pts[4], hands[1].pts[8]], np.float32))
                if quad_area(quad) > 0.04 * w * h:
                    self.hold += dt
                    fx = pixel(frame, {"cell": 16})
                    mk = np.zeros((h, w), np.uint8)
                    cv2.fillConvexPoly(mk, quad.astype(np.int32), 255, cv2.LINE_AA)
                    cv2.copyTo(fx, mk, frame)
                    m = "Parfait, garde la pose…"
                else:
                    self.hold = max(0.0, self.hold - dt)
                    m = "Agrandis le cadre"
                ui.draw_quad(frame, quad)
                self.disp = ui.approach(self.disp, self.hold / 1.5, dt, 14)
                ui.arc(frame, quad.mean(axis=0), ui.u(46), self.disp, ui.GREEN, 5)
            else:
                self.hold = max(0.0, self.hold - dt)
            sub = self.msg(m)
            if self.hold >= 1.5:
                self._set("done")

        elif p == "done":
            ui.dim(frame, 0.5)
            c = (w // 2, h // 2 - ui.u(30))
            if not self._burst:
                self._burst = True
                for k in range(6):
                    self.particles.emit(c, 30, hue=k / 6, speed=320, direction=-math.pi / 2, spread=math.pi * 1.6)
            ui.arc(frame, c, ui.u(90), ui.ease_out(e / 0.5), ui.GREEN, 6)
            ui.checkmark(frame, c, ui.u(110), (e - 0.3) / 0.5, ui.GREEN, 10)
            a = ui.ease_out((e - 0.4) / 0.5)
            T.draw(frame, "C'est prêt !", (w // 2, h // 2 + ui.u(110)), 44, ui.WHITE, "mm", bold=True, alpha=a)
            T.draw(frame, "Pince pouce + index pour changer d'effet  ·  TAB pour l'aide",
                   (w // 2, h // 2 + ui.u(160)), 22, ui.WHITE, "mm", alpha=0.85 * a)
            if e > 2.8:
                self.particles.update(dt)
                self.particles.draw(frame)
                return "done", self._build_profile()

        self.particles.update(dt)
        self.particles.draw(frame)
        _titles(T, frame, title, sub, self.t_phase)
        if p != "done":
            ui.step_dots(frame, (w // 2, h - ui.u(60)), self._step_index(), 4)
            T.draw(frame, hint or "ESPACE : passer l'étape   ·   Q : quitter", (w // 2, h - ui.u(30)), 18,
                   ui.WHITE, "mm", alpha=0.6)
        self.fader.apply(frame)
        self.last = frame
        return None

    def _build_profile(self):
        prof = dict(self.profile)
        open_r = float(np.median(self.open_samples)) if self.open_samples else 1.2
        pinch_r = float(np.median(self.pinch_samples)) if self.pinch_samples else 0.2
        prof.update(open_ratio=open_r, pinch_ratio=pinch_r,
                    pinch_threshold=float(np.clip(pinch_r + 0.3 * (open_r - pinch_r), 0.2, 0.42)),
                    palm_px=float(np.median(self.palms)) if self.palms else None,
                    created=time.strftime("%Y-%m-%d %H:%M:%S"), version=2)
        return prof


# =========================================================================== déverrouillage
class Unlock:
    """Initialisation à chaque lancement : scan du visage (comparé au profil) puis repérage des mains."""

    SCAN_TIME = 2.5

    def __init__(self, text: ui.Text, profile: dict):
        self.text = text
        self.ref = profile.get("face_signature")
        self.base = profile.get("face_baseline", [0.0, 0.0])
        self.sigs = []
        self.scan_t = 0.0
        self.score = None
        self.fader = _Fader()
        self.last = None
        self.msg = _StableText()
        self.disp = 0.0
        self.hand_hold = {}
        self.first_ok = None
        self.particles = ui.Particles()
        self.phase = None
        self._set("scan" if self.ref else "hands")

    def _set(self, phase):
        if self.phase is not None:
            self.fader.start(self.last)
        self.phase = phase
        self.t_phase = time.monotonic()
        self.msg = _StableText()

    @property
    def needs_face(self):
        return self.phase == "scan"

    @property
    def needs_hands(self):
        return self.phase == "hands"

    needs_seg = False

    def key(self, k):
        if k in (ord("q"), ESC):
            return "quit"
        if k == ord("r"):
            return "recalibrate"
        if k == SPACE:
            if self.phase in ("scan", "fail", "ok"):
                self._set("hands")
                return None
            return "skip"
        if k == ord("t") and self.phase == "fail":
            self.sigs, self.scan_t = [], 0.0
            self._set("scan")
        return None

    def update(self, frame, tracker, dt):
        h, w = frame.shape[:2]
        center, r = _circle_geom(frame)
        T = self.text
        e = time.monotonic() - self.t_phase
        r_in, r_out = r + ui.u(18), r + ui.u(32)
        title = sub = ""
        if self.phase in ("scan", "ok", "fail"):
            ui.focus_circle(frame, center, r, dim=0.22)
        else:
            ui.dim(frame, 0.55)

        if self.phase == "scan":
            face = tracker.face()
            title = "Initialisation"
            ok, m = _face_hint(face, center, r)
            frontal = ok and abs(face.yaw - self.base[0]) < 0.18 and abs(face.pitch - self.base[1]) < 0.18
            if frontal:
                self.sigs.append(face.signature)
                self.scan_t += dt
                m = "Analyse du visage…"
            elif ok:
                m = "Regarde l'écran"
            sub = self.msg(m)
            prog = min(1.0, self.scan_t / self.SCAN_TIME)
            self.disp = ui.approach(self.disp, prog, dt, 10)
            # graduations : une zone tourne pendant la recherche, l'anneau se remplit avec l'analyse
            lv = np.full(N_TICKS, 0.2, np.float32)
            filled = int(self.disp * N_TICKS)
            lv[:filled] = 1.0
            head = (e * 45) % N_TICKS
            for k in range(14):
                j = int(head - k) % N_TICKS
                lv[j] = max(lv[j], 0.8 * (1 - k / 14))
            ui.ticks_ring(frame, center, r_in, r_out, lv)
            y = int(center[1] - r + (math.sin(e * 2.6) * 0.5 + 0.5) * 2 * r)
            half = int(math.sqrt(max(r * r - (y - center[1]) ** 2, 0)))
            over = frame.copy()
            cv2.line(over, (center[0] - half, y), (center[0] + half, y), ui.GREEN, ui.u(3), cv2.LINE_AA)
            cv2.addWeighted(over, 0.7, frame, 0.3, 0, frame)
            _face_mesh(frame, face, center, r, e)
            T.draw(frame, f"{int(round(self.disp * 100))} %", (w // 2, ui.u(136)), 22, ui.GREEN, "mt", bold=True)
            if e > 10 and not self.sigs:
                T.draw(frame, "Aucun visage détecté — ESPACE pour passer", (w // 2, ui.u(170)), 20,
                       ui.WHITE, "mt", alpha=0.7)
            if prog >= 1.0:
                self.score = signature_distance(np.median(np.array(self.sigs[-60:]), axis=0), self.ref)
                print(f"[init] score visage = {self.score:.4f} (seuil {FACE_MATCH_THRESHOLD})")
                self._set("ok" if self.score < FACE_MATCH_THRESHOLD else "fail")

        elif self.phase == "ok":
            title = "Visage reconnu"
            ui.ticks_ring(frame, center, r_in, r_out, np.ones(N_TICKS))
            ui.dim(frame, 1.0 - 0.45 * ui.ease_out(e / 0.3))
            ui.checkmark(frame, center, r * 0.8, e / 0.5, ui.GREEN, 10)
            if e > 1.2:
                self._set("hands")

        elif self.phase == "fail":
            title = "Visage non reconnu"
            sub = "T : réessayer   ·   ESPACE : continuer   ·   R : réenregistrer   ·   Q : quitter"
            shake = int(math.sin(e * 40) * ui.u(14) * max(0.0, 1 - e / 0.5))
            ui.ticks_ring(frame, (center[0] + shake, center[1]), r_in, r_out, np.ones(N_TICKS), color_on=ui.RED)

        elif self.phase == "hands":
            title = "Repérage des mains"
            hands = tracker.hands()
            done = []
            seen = set()
            for i, hd in enumerate(hands):
                key = hd.label + str(i)
                seen.add(key)
                ui.draw_hand(frame, hd)
                t = self.hand_hold.get(key, 0.0) + dt
                self.hand_hold[key] = t
                c = hd.pts[9]
                ui.arc(frame, c, hd.palm * 1.25, min(1.0, t / 0.8), ui.GREEN, 6)
                if t >= 0.8:
                    done.append(key)
                    ui.checkmark(frame, c, hd.palm * 0.8, (t - 0.8) / 0.35, ui.GREEN, 7)
                    if t - dt < 0.8:
                        self.particles.emit(c, 40, hue=0.33, speed=260)
            for key in list(self.hand_hold):
                if key not in seen:
                    self.hand_hold[key] = max(0.0, self.hand_hold[key] - 2 * dt)
            if len(done) >= 2:
                sub = "Les deux mains sont repérées"
            elif done:
                sub = self.msg("Une main repérée — montre l'autre (ou attends)")
            else:
                sub = self.msg("Montre tes mains à la caméra")
            if done and self.first_ok is None:
                self.first_ok = time.monotonic()
            if not done:
                self.first_ok = None
            if (len(done) >= 2 and self.first_ok and time.monotonic() - self.first_ok > 0.5) or \
                    (self.first_ok and time.monotonic() - self.first_ok > 2.5):
                self._set("ready")
            T.draw(frame, "ESPACE : passer   ·   Q : quitter", (w // 2, h - ui.u(30)), 18, ui.WHITE, "mm", alpha=0.6)

        elif self.phase == "ready":
            c = (w // 2, h // 2 - ui.u(20))
            ui.arc(frame, c, ui.u(80), ui.ease_out(e / 0.4), ui.GREEN, 6)
            ui.checkmark(frame, c, ui.u(96), (e - 0.2) / 0.4, ui.GREEN, 9)
            T.draw(frame, "Détection prête", (w // 2, h // 2 + ui.u(100)), 40, ui.WHITE, "mm", bold=True,
                   alpha=ui.ease_out((e - 0.3) / 0.4))
            if e > 1.3:
                self.last = frame
                return "done"

        self.particles.update(dt)
        self.particles.draw(frame)
        _titles(T, frame, title, sub, self.t_phase)
        if self.score is not None and self.phase in ("ok", "fail"):
            T.draw(frame, f"score {self.score:.3f} / seuil {FACE_MATCH_THRESHOLD}", (w - ui.u(16), h - ui.u(16)),
                   16, ui.WHITE, "rb", alpha=0.5)
        self.fader.apply(frame)
        self.last = frame
        return None
