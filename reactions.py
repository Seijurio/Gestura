"""Réactions déclenchées par les gestes (dans l'esprit des réactions FaceTime) :
👍 confettis · 🤘 lasers · 🫶 cœurs · 👏 onde de choc · ✨ explosion.
"""
import math
import random
import time

import cv2
import numpy as np

import icons
import ui

CONFETTI_COLORS = [(70, 60, 255), (40, 200, 255), (90, 255, 110), (255, 200, 60), (255, 80, 200), (255, 255, 255)]


class Reactions:
    def __init__(self):
        self.sprites = []      # dict: char, x, y, vx, vy, size, t0, life, wob, pop
        self.confetti = np.zeros((0, 9), np.float32)  # x y vx vy life angle spin color size
        self.emitters = []     # (char, t_end, rate, acc)
        self.lasers_until = 0.0
        self.waves = []        # (x, y, t0)
        self.flash_t = -10.0

    # ------------------------------------------------------------------ déclencheurs
    def pop(self, char, pos, size=150, life=1.6, rise=60):
        self.sprites.append(dict(char=char, x=float(pos[0]), y=float(pos[1]), vx=0.0, vy=-rise * ui.S,
                                 size=size * ui.S, t0=time.monotonic(), life=life, wob=random.random() * 6,
                                 pop=True))

    def confetti_burst(self, pos, n=160):
        a = np.random.uniform(-math.pi, 0, n)
        sp = np.random.uniform(300, 900, n) * ui.S
        c = np.stack([np.full(n, pos[0]), np.full(n, pos[1]), np.cos(a) * sp, np.sin(a) * sp,
                      np.random.uniform(1.5, 2.8, n), np.random.uniform(0, 6.3, n), np.random.uniform(-9, 9, n),
                      np.random.randint(0, len(CONFETTI_COLORS), n), np.random.uniform(6, 12, n) * ui.S],
                     axis=1).astype(np.float32)
        self.confetti = np.concatenate([self.confetti, c])[-1200:]

    def thumbs(self, pos):
        self.pop("👍", pos, 170)
        self.confetti_burst(pos)

    def hearts(self, pos):
        self.pop("❤️", pos, 160, 1.4)
        self.emitters.append(["❤️", time.monotonic() + 2.2, 14.0, 0.0])

    def rock(self, pos):
        self.pop("🤘", pos, 150)
        self.lasers_until = time.monotonic() + 3.5

    def clap(self, pos):
        self.pop("👏", pos, 140, 1.0)
        self.waves.append((float(pos[0]), float(pos[1]), time.monotonic()))
        self.flash_t = time.monotonic()

    def sparkle(self, pos, char="✨"):
        self.pop(char, pos, 110, 1.0)

    @property
    def busy(self):
        return bool(self.sprites or len(self.confetti) or self.emitters or self.waves
                    or time.monotonic() < self.lasers_until)

    # ------------------------------------------------------------------ animation
    def update_draw(self, frame, dt):
        h, w = frame.shape[:2]
        now = time.monotonic()

        if now < self.lasers_until:
            self._lasers(frame, now)

        for wx, wy, t0 in list(self.waves):
            e = (now - t0) / 0.7
            if e >= 1:
                self.waves.remove((wx, wy, t0))
                continue
            r = int(ui.ease_out(e) * 0.6 * w)
            over = frame.copy()
            cv2.circle(over, (int(wx), int(wy)), r, (255, 255, 255), max(2, int(ui.u(14) * (1 - e))), cv2.LINE_AA)
            cv2.addWeighted(over, 0.7 * (1 - e), frame, 1 - 0.7 * (1 - e), 0, frame)
        fe = now - self.flash_t
        if fe < 0.15:
            cv2.addWeighted(frame, 1 - 0.35 * (1 - fe / 0.15), np.full_like(frame, 255), 0.35 * (1 - fe / 0.15), 0,
                            frame)

        for em in list(self.emitters):
            char, t_end, rate, acc = em
            if now > t_end:
                self.emitters.remove(em)
                continue
            em[3] += rate * dt
            while em[3] >= 1:
                em[3] -= 1
                x = random.uniform(0.1, 0.9) * w
                self.sprites.append(dict(char=char, x=x, y=h + ui.u(40), vx=random.uniform(-30, 30) * ui.S,
                                         vy=-random.uniform(220, 420) * ui.S, size=random.uniform(50, 110) * ui.S,
                                         t0=now, life=random.uniform(2.2, 3.2), wob=random.random() * 6, pop=False))

        if len(self.confetti):
            c = self.confetti
            c[:, 0] += c[:, 2] * dt
            c[:, 1] += c[:, 3] * dt
            c[:, 3] += 900 * ui.S * dt
            c[:, 2] *= 0.98
            c[:, 3] = np.minimum(c[:, 3], 260 * ui.S)
            c[:, 4] -= dt
            c[:, 5] += c[:, 6] * dt
            self.confetti = c[(c[:, 4] > 0) & (c[:, 1] < h + 20)]
            for x, y, _, _, life, ang, _, ci, sz in self.confetti:
                ca, sa = math.cos(ang), math.sin(ang)
                sx, sy = sz, sz * 0.45 * abs(math.cos(ang * 1.7))
                pts = np.array([[-sx * ca + sy * sa, -sx * sa - sy * ca], [sx * ca + sy * sa, sx * sa - sy * ca],
                                [sx * ca - sy * sa, sx * sa + sy * ca], [-sx * ca - sy * sa, -sx * sa + sy * ca]])
                cv2.fillConvexPoly(frame, (pts + (x, y)).astype(np.int32), CONFETTI_COLORS[int(ci)], cv2.LINE_AA)

        for s in list(self.sprites):
            e = (now - s["t0"]) / s["life"]
            if e >= 1:
                self.sprites.remove(s)
                continue
            s["x"] += (s["vx"] + math.sin(now * 3 + s["wob"]) * 40 * ui.S) * dt
            s["y"] += s["vy"] * dt
            if s["pop"]:  # rebond d'apparition
                k = (now - s["t0"]) / 0.35
                scale = 1.0 + 0.35 * math.sin(min(k, 1) * math.pi) if k < 1 else 1.0
                scale *= ui.ease_out(min(1.0, k * 2.5))
            else:
                scale = 1.0
            op = 1.0 if e < 0.7 else 1 - (e - 0.7) / 0.3
            size = int(s["size"] * scale / 4) * 4
            if size >= 8:
                ui.blit(frame, icons.get(s["char"], size), (s["x"], s["y"]), op)

    def _lasers(self, frame, now):
        h, w = frame.shape[:2]
        layer = np.zeros((h // 2, w // 2, 3), np.uint8)
        sw, sh = w // 2, h // 2
        for k, (ox, oy) in enumerate(((0, sh), (sw, sh), (sw // 2, 0), (0, 0), (sw, 0))):
            for j in range(3):
                a = math.sin(now * (1.3 + 0.4 * k) + j * 2.1 + k) * 0.9 + (math.pi / 2 if oy == 0 else -math.pi / 2)
                end = (int(ox + math.cos(a) * sw * 1.5), int(oy + math.sin(a) * sw * 1.5))
                col = ui.hsv_color((now * 0.4 + k * 0.2 + j * 0.07) % 1.0, 0.9, 1.0)
                cv2.line(layer, (ox, oy), end, col, max(1, ui.u(3)), cv2.LINE_AA)
        glow = ui.soft_blur(layer, ui.u(6))
        layer = cv2.addWeighted(layer, 1.0, glow, 2.0, 0)
        cv2.add(frame, cv2.resize(layer, (w, h)), frame)
