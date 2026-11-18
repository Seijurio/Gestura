"""Menu circulaire : main ouverte immobile 1 s → le menu s'ouvre autour de ta main.
Déplace la main vers un effet (aperçu en direct au centre), ferme le poing ou pince pour choisir.
"""
import math
import time

import cv2
import numpy as np

import ui


class RadialMenu:
    def __init__(self, effects, current, center, frame_size):
        w, h = frame_size
        self.effects = effects
        self.n = len(effects)
        self.R = 0.30 * h
        self.c = np.array([min(max(center[0], self.R * 1.3), w - self.R * 1.3),
                           min(max(center[1], self.R * 1.3), h - self.R * 1.3)], np.float32)
        self.origin = np.array(center, np.float32)
        self.sel = current
        self.t0 = time.monotonic()
        self.lost = 0.0
        self.anim = 0.0
        self.state = {}
        self.ptr = self.c.copy()

    def update(self, hand, pose, dt):
        """Renvoie ('select', i), ('cancel',) ou None."""
        now = time.monotonic()
        self.anim = ui.approach(self.anim, 1.0, dt, 9)
        if hand is None:
            self.lost += dt
            return ("cancel",) if self.lost > 0.8 else None
        self.lost = 0.0
        # le déplacement de la main par rapport au point d'ouverture pilote le pointeur
        off = (hand.pts[9] - self.origin) * 1.6
        self.ptr = ui.approach(self.ptr, self.c + off, dt, 18)
        v = self.ptr - self.c
        if np.hypot(*v) > 0.3 * self.R:
            a = math.atan2(v[1], v[0]) + math.pi / 2
            self.sel = int(round(a / (2 * math.pi) * self.n)) % self.n
        if now - self.t0 > 0.5 and pose in ("fist", "pinch"):
            return ("select", self.sel)
        if now - self.t0 > 10:
            return ("cancel",)
        return None

    def draw(self, frame, cell, color, t):
        h, w = frame.shape[:2]
        T = ui.Text_default()
        k = ui.ease_out(self.anim)
        R = self.R * (0.7 + 0.3 * k)
        cx, cy = int(self.c[0]), int(self.c[1])
        orig = frame.copy()
        # fond flouté dans le disque
        m = ui.feather_mask(h, w, lambda mk, s: cv2.circle(mk, (int(cx * s), int(cy * s)), int(R * 1.28 * s), 255, -1),
                            ui.u(6))
        bg = cv2.convertScaleAbs(ui.soft_blur(frame, 14 * ui.S), alpha=0.45)
        frame[:] = cv2.blendLinear(bg, frame, m * k, 1.0 - m * k)
        # aperçu live de l'effet survolé
        pr = int(R * 0.48)
        x0, y0 = max(0, cx - pr), max(0, cy - pr)
        x1, y1 = min(w, cx + pr), min(h, cy + pr)
        crop = orig[y0:y1, x0:x1]
        if crop.size:
            small = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_AREA)
            name, fn, _ = self.effects[self.sel]
            ctx = {"cell": cell, "t": t, "dt": 0.033, "color": color,
                   "quad": np.float32([[0, 0], [360, 0], [360, 360], [0, 360]]), "center": np.float32([180, 180]),
                   "prev": None, "history": None, "mask": None, "bg": None, "state": self.state}
            try:
                prev = fn(small, ctx)
            except Exception:  # noqa: BLE001 — un aperçu raté ne doit jamais faire planter le menu
                prev = small
            prev = cv2.resize(prev, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
            pm = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.circle(pm, (cx - x0, cy - y0), pr, 255, -1, cv2.LINE_AA)
            cv2.copyTo(prev, pm, frame[y0:y1, x0:x1])
        cv2.circle(frame, (cx, cy), pr, ui.WHITE, max(1, ui.u(2)), cv2.LINE_AA)
        # libellés autour
        for i, (name, _, _) in enumerate(self.effects):
            a = 2 * math.pi * i / self.n - math.pi / 2
            px, py = cx + math.cos(a) * R, cy + math.sin(a) * R
            cur = i == self.sel
            if cur:
                tw, th = T.size(name, 19, True)
                cv2.ellipse(frame, (int(px), int(py)), (int(tw / 2 + ui.u(12)), int(th / 2 + ui.u(8))), 0, 0, 360,
                            ui.GREEN, -1, cv2.LINE_AA)
                T.draw(frame, name, (px, py), 19, (20, 20, 20), "mm", bold=True, alpha=k)
            else:
                T.draw(frame, name, (px, py), 15, ui.WHITE, "mm", stroke=2, alpha=0.85 * k)
        # pointeur
        cv2.line(frame, (cx, cy), (int(self.ptr[0]), int(self.ptr[1])), ui.WHITE, max(1, ui.u(1.5)), cv2.LINE_AA)
        cv2.circle(frame, (int(self.ptr[0]), int(self.ptr[1])), ui.u(9), ui.GREEN, -1, cv2.LINE_AA)
        T.draw(frame, "Bouge la main vers un effet · ferme le poing pour choisir", (cx, cy + R * 1.28 + ui.u(26)), 18,
               ui.WHITE, "mm", stroke=2, alpha=k)
