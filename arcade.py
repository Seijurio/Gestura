"""Salle d'arcade + 3 jeux supplémentaires.

Arcade      — choix du jeu : pointe une carte avec l'index (0,8 s), ou touches 1-5, ou ← → + Entrée.
SliceGame   — « Tranche » : coupe les fruits lancés en l'air d'un geste vif de l'index, évite les bombes.
SimonGame   — « Jacques a dit » : fais le geste demandé… seulement si Jacques l'a dit !
DodgeGame   — « Esquive » : ta tête pilote, évite les météores, attrape les étoiles.
"""
import collections
import math
import random
import time

import cv2
import numpy as np

import audio as A
import icons
import ui
from games import BubbleGame, DanceGame, _Game
from gestures import EMOJI, GestureEngine

ESC, SPACE, ENTER = 27, 32, 13


# =========================================================================== TRANCHE
FRUITS = list(icons.JUICE.items())  # fruits dessinés par icons.py, avec la couleur de leur jus


def _rot_sprite(sprite, ang):
    bgr, a = sprite
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return (cv2.warpAffine(bgr, M, (w, h)), cv2.warpAffine(a, M, (w, h))[..., None])


class SliceGame(_Game):
    DURATION = 60.0

    def __init__(self, text, audio, records):
        super().__init__(text, audio, records, "tranche")
        self.items = []     # fruits et bombes en vol
        self.halves = []    # moitiés de fruits coupés
        self.blades = {}    # clé main -> deque[(t, p)]
        self.lives = 3
        self.spawn_t = 0.0
        self.t_start = None
        self.sliced = 0
        self.best_combo = 0
        self.combo = collections.defaultdict(list)
        self.stain = None
        self.shake = 0.0
        self.flash = 0.0
        self._seq = -1
        self._last_whoosh = 0.0

    # ------------------------------------------------------------------ lames (bout des index)
    def _update_blades(self, tracker, now):
        seq = tracker.hands_seq
        if seq == self._seq:
            return
        self._seq = seq
        seen = set()
        for i, hd in enumerate(tracker.hands()):
            k = hd.label + str(i)
            seen.add(k)
            dq = self.blades.setdefault(k, collections.deque(maxlen=12))
            dq.append((now, hd.pts[8].astype(np.float32)))
        for k in list(self.blades):
            if k not in seen:
                del self.blades[k]

    def _blade_segments(self, now, w):
        """Segments récents des lames rapides : [(clé, p0, p1, vitesse)]"""
        out = []
        for k, dq in self.blades.items():
            pts = [(t, p) for t, p in dq if now - t < 0.12]
            for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
                sp = float(np.linalg.norm(p1 - p0)) / max(1e-3, t1 - t0) / w
                if sp > 0.9:
                    out.append((k, p0, p1, sp))
        return out

    # ------------------------------------------------------------------ boucle
    def update(self, frame, tracker, dt):
        h, w = frame.shape[:2]
        self.w, self.h = w, h
        T = self.text
        now = time.monotonic()
        e = now - self.t_phase
        self._update_blades(tracker, now)
        if self.phase == "wait":
            ui.dim(frame, 0.6)
            T.draw(frame, "TRANCHE", (w // 2, ui.u(60)), 64, ui.WHITE, "mt", bold=True, stroke=3)
            T.draw(frame, "Coupe les fruits d'un geste vif de l'index · ne touche pas les bombes (3 vies)",
                   (w // 2, ui.u(140)), 24, ui.WHITE, "mt", stroke=2)
            hands = tracker.hands()
            self.hold = self.hold + dt if hands else max(0.0, self.hold - dt)
            T.draw(frame, "Montre tes mains pour commencer" if not hands else "C'est parti…", (w // 2, h // 2), 30,
                   ui.HUD_GREEN if hands else ui.WHITE, "mm", bold=True, stroke=2)
            ui.arc(frame, (w // 2, h // 2 + ui.u(80)), ui.u(30), self.hold / 0.8, ui.GREEN, 5)
            if self.hold >= 0.8:
                self._set("countdown")
        elif self.phase == "countdown":
            left = 3 - e
            if left <= 0:
                self._set("play")
                self.t_start = now
                self.start_music(2, 0.28)
            else:
                T.draw(frame, str(int(left) + 1), (w // 2, h // 2), 140, ui.WHITE, "mm", bold=True, stroke=4,
                       alpha=0.4 + 0.6 * (left - int(left)))
        elif self.phase == "play":
            self._play(frame, dt, now)
        if self.phase == "end":
            self._end_screen(frame, [("Score", str(self.score)), ("Fruits tranchés", str(self.sliced)),
                                     ("Meilleur combo", str(self.best_combo)),
                                     ("Record", str(self.records.get("tranche", 0)))])
        self._draw_blades(frame, now)
        self.particles.update(dt)
        self.particles.draw(frame)
        self._draw_pops(frame)
        return None

    def _spawn(self, w, h, el):
        n = random.choices([1, 2, 3, 4], [0.45, 0.3, 0.17, 0.08 + el * 0.002])[0]
        g = 1.9 * h
        for _ in range(n):
            bomb = random.random() < 0.12 + el * 0.0025
            x = random.uniform(0.15, 0.85) * w
            peak = random.uniform(0.45, 0.82) * h
            vy = -math.sqrt(2 * g * peak)
            vx = (w / 2 - x) * random.uniform(0.3, 0.9) / 1.2
            char, col = ("bomb", (40, 40, 40)) if bomb else random.choice(FRUITS)
            self.items.append({"char": char, "col": col, "bomb": bomb, "x": x, "y": h + ui.u(60), "vx": vx,
                               "vy": vy, "ang": random.uniform(0, 360), "spin": random.uniform(-220, 220),
                               "r": ui.u(58 if not bomb else 52), "g": g})

    def _play(self, frame, dt, now):
        h, w = frame.shape[:2]
        T = self.text
        el = now - self.t_start
        left = self.DURATION - el
        if now >= self.spawn_t:
            self._spawn(w, h, el)
            self.spawn_t = now + max(0.55, 1.5 - el * 0.015) * random.uniform(0.8, 1.2)
        # taches de jus qui s'estompent
        if self.stain is not None:
            self.stain = cv2.convertScaleAbs(self.stain, alpha=0.985)
            frame[:] = cv2.subtract(frame, cv2.resize(self.stain, (w, h)))
        segs = self._blade_segments(now, w)
        if segs and now - self._last_whoosh > 0.25:
            self._last_whoosh = now
            self._sfx(A.sfx_slice(), 0.25)
        for it in self.items:
            it["vy"] += it["g"] * dt
            it["x"] += it["vx"] * dt
            it["y"] += it["vy"] * dt
            it["ang"] += it["spin"] * dt
            for k, p0, p1, sp in segs:
                d = p1 - p0
                L = float(np.dot(d, d)) + 1e-6
                tt = float(np.clip(np.dot(np.array([it["x"], it["y"]]) - p0, d) / L, 0, 1))
                if np.linalg.norm(p0 + d * tt - (it["x"], it["y"])) < it["r"] * 0.9:
                    self._slice(it, d, k, now)
                    break
        self.items = [it for it in self.items if not it.get("dead") and it["y"] < h + ui.u(140)]
        # moitiés
        for hv in self.halves:
            hv["vy"] += hv["g"] * dt
            hv["x"] += hv["vx"] * dt
            hv["y"] += hv["vy"] * dt
            hv["ang"] += hv["spin"] * dt
        self.halves = [hv for hv in self.halves if now - hv["t0"] < 1.6 and hv["y"] < h + ui.u(150)]
        for hv in self.halves:
            ui.blit(frame, _rot_sprite(hv["sprite"], hv["ang"]), (hv["x"], hv["y"]), 1 - (now - hv["t0"]) / 1.6)
        for it in self.items:
            ui.blit(frame, _rot_sprite(icons.get(it["char"], it["r"] * 2), it["ang"]), (it["x"], it["y"]))
            if it["bomb"] and random.random() < 0.5:
                a = math.radians(-it["ang"] - 45)
                self.particles.emit((it["x"] + math.cos(a) * it["r"] * 0.7, it["y"] + math.sin(a) * it["r"] * 0.7),
                                    1, hue=0.1, speed=80)
        if self.shake > 0:
            self.shake = max(0.0, self.shake - dt * 2.5)
            frame[:] = np.roll(frame, int(math.sin(now * 70) * ui.u(22) * self.shake), axis=1)
        if self.flash > 0:
            self.flash = max(0.0, self.flash - dt * 3)
            cv2.addWeighted(frame, 1 - self.flash * 0.8, np.full_like(frame, 255), self.flash * 0.8, 0, frame)
        # interface
        cv2.rectangle(frame, (0, 0), (int(w * max(0.0, left / self.DURATION)), ui.u(8)),
                      ui.HUD_GREEN if left > 10 else ui.RED, -1)
        T.draw(frame, str(self.score), (ui.u(24), ui.u(22)), 44, ui.WHITE, "lt", bold=True, stroke=3)
        T.draw(frame, f"{int(math.ceil(max(0.0, left)))} s", (w - ui.u(24), ui.u(22)), 36, ui.WHITE, "rt", bold=True,
               stroke=3)
        for i in range(3):
            ui.blit(frame, icons.get("❤️" if i < self.lives else "🖤", ui.u(40)),
                    (w // 2 + (i - 1) * ui.u(52), ui.u(44)), 1.0 if i < self.lives else 0.5)
        if left <= 0 or self.lives <= 0:
            self._finish()

    def _slice(self, it, d, key, now):
        it["dead"] = True
        x, y = it["x"], it["y"]
        if it["bomb"]:
            self.lives -= 1
            self.shake, self.flash = 1.0, 1.0
            self.particles.emit((x, y), 120, hue=0.06, speed=700)
            self._popup("BOUM !", (x, y), ui.RED, 60)
            self._sfx(A.sfx_boom(), 0.8)
            return
        self.sliced += 1
        # combo : plusieurs fruits d'un même geste
        lst = [t for t in self.combo[key] if now - t < 0.35] + [now]
        self.combo[key] = lst
        pts = 10
        if len(lst) >= 3:
            pts += 10 * len(lst)
            self._popup(f"COMBO x{len(lst)} !", (x, y - ui.u(60)), (80, 220, 255), 46)
            self.best_combo = max(self.best_combo, len(lst))
        self.score += pts
        self._popup(f"+{pts}", (x, y), ui.WHITE, 34)
        # deux moitiés coupées le long du geste
        bgr, a = _rot_sprite(icons.get(it["char"], it["r"] * 2), it["ang"])
        hh, ww = a.shape[:2]
        n = np.array([-d[1], d[0]], np.float32)
        n /= np.linalg.norm(n) + 1e-6
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        side = ((xx - ww / 2) * n[0] + (yy - hh / 2) * n[1]) > 0
        for sgn in (1, -1):
            m = (side if sgn > 0 else ~side).astype(np.float32)[..., None]
            self.halves.append({"sprite": (bgr, a * m), "x": x, "y": y, "vx": it["vx"] * 0.5 + sgn * n[0] * ui.u(240),
                                "vy": min(it["vy"], 0) * 0.3 + sgn * n[1] * ui.u(240), "ang": 0.0,
                                "spin": sgn * random.uniform(90, 200), "g": it["g"], "t0": now})
        # jus
        hue = cv2.cvtColor(np.uint8([[it["col"]]]), cv2.COLOR_BGR2HSV)[0, 0, 0] / 180.0
        self.particles.emit((x, y), 45, hue=hue, speed=420)
        if self.stain is None:
            self.stain = np.zeros((self.h // 4, self.w // 4, 3), np.uint8)
        col = tuple(int(255 - c) for c in it["col"])
        for _ in range(6):
            cv2.circle(self.stain, (int(x / 4 + random.uniform(-12, 12)), int(y / 4 + random.uniform(-12, 12))),
                       random.randint(3, 10), tuple(int(c * 0.5) for c in col), -1, cv2.LINE_AA)
        self._sfx(A.sfx_pop(0.6), 0.5)

    def _draw_blades(self, frame, now):
        h, w = frame.shape[:2]
        layer = np.zeros((h // 2, w // 2, 3), np.uint8)
        for k, dq in self.blades.items():
            pts = [(t, p) for t, p in dq if now - t < 0.15]
            for i, ((t0, p0), (t1, p1)) in enumerate(zip(pts, pts[1:])):
                sp = float(np.linalg.norm(p1 - p0)) / max(1e-3, t1 - t0) / w
                th = max(1, int(ui.u(5) * (i + 1) / len(pts)))
                col = (255, 255, 255) if sp > 0.9 else (180, 180, 180)
                cv2.line(layer, tuple((p0 / 2).astype(int)), tuple((p1 / 2).astype(int)), col, th, cv2.LINE_AA)
        glow = ui.soft_blur(layer, ui.u(4))
        layer = cv2.add(layer, cv2.merge([glow[..., 0], glow[..., 1], cv2.convertScaleAbs(glow[..., 2], alpha=0.6)]))
        cv2.add(frame, cv2.resize(layer, (w, h)), frame)


# =========================================================================== JACQUES A DIT
SIMON_POSES = [("open", "main ouverte"), ("fist", "poing fermé"), ("point", "index levé"), ("peace", "fais un V"),
               ("rock", "signe rock"), ("thumbs_up", "pouce levé"), ("vulcan", "salut vulcain")]


class SimonGame(_Game):
    def __init__(self, text, audio, records):
        super().__init__(text, audio, records, "jacques")
        self.engine = GestureEngine(0.42, audio)
        self.lives = 3
        self.round = 0
        self.streak = self.best_streak = 0
        self.cmd = None
        self.state = "ready"
        self.state_t = time.monotonic()
        self.done_t = None
        self.result = None
        self.last_pose = None
        self.clap_t = -10.0

    def _poses(self):
        return [p for p, hd in self.engine.poses().values() if hd is not None]

    def _new_cmd(self):
        choices = SIMON_POSES[:]
        if self.audio is not None and self.audio.mic_ok:
            choices.append(("clap", "applaudis"))
        choices = [c for c in choices if self.cmd is None or c[0] != self.cmd["pose"]]
        pose, label = random.choice(choices)
        simon = random.random() < 0.7 or self.round < 2
        window = max(1.4, 3.2 - self.round * 0.09)
        self.cmd = {"pose": pose, "label": label, "simon": simon, "window": window, "t": time.monotonic(),
                    "held": 0.0, "already": pose in self._poses()}
        self.round += 1

    def update(self, frame, tracker, dt):
        h, w = frame.shape[:2]
        self.w, self.h = w, h
        T = self.text
        now = time.monotonic()
        hands = tracker.hands()
        for ev in self.engine.update(hands, tracker.hands_seq, now, (w, h), {"clap"}):
            if ev[0] == "clap":
                self.clap_t = now
        poses = self._poses()
        ui.dim(frame, 0.55)
        e = now - self.t_phase
        if self.phase == "wait":
            T.draw(frame, "JACQUES A DIT", (w // 2, ui.u(60)), 60, ui.WHITE, "mt", bold=True, stroke=3)
            T.draw(frame, "Fais le geste demandé… mais SEULEMENT si la consigne commence par « Jacques a dit »",
                   (w // 2, ui.u(140)), 22, ui.WHITE, "mt", stroke=2)
            for k, (p, _) in enumerate(SIMON_POSES):
                ui.blit(frame, icons.get(EMOJI[p], ui.u(64)), (w // 2 + (k - (len(SIMON_POSES) - 1) / 2) * ui.u(100), h // 2 - ui.u(40)))
            self.hold = self.hold + dt if hands else max(0.0, self.hold - dt)
            T.draw(frame, "Montre une main pour commencer" if not hands else "C'est parti…",
                   (w // 2, h // 2 + ui.u(70)), 28, ui.HUD_GREEN if hands else ui.WHITE, "mm", bold=True, stroke=2)
            ui.arc(frame, (w // 2, h // 2 + ui.u(140)), ui.u(30), self.hold / 0.8, ui.GREEN, 5)
            if self.hold >= 0.8:
                self._set("countdown")
        elif self.phase == "countdown":
            left = 3 - e
            if left <= 0:
                self._set("play")
                self.state, self.state_t = "ready", now
            else:
                T.draw(frame, str(int(left) + 1), (w // 2, h // 2), 140, ui.WHITE, "mm", bold=True, stroke=4,
                       alpha=0.4 + 0.6 * (left - int(left)))
        elif self.phase == "play":
            self._play(frame, poses, now, dt)
        if self.phase == "end":
            self._end_screen(frame, [("Score", str(self.score)), ("Manches", str(self.round)),
                                     ("Meilleure série", str(self.best_streak)),
                                     ("Record", str(self.records.get("jacques", 0)))])
        # ce que la caméra voit
        if self.phase == "play":
            shown = [p for p in poses if p in EMOJI]
            T.draw(frame, "je vois :", (w - ui.u(200), h - ui.u(60)), 18, ui.WHITE, "rm", alpha=0.7)
            for k, p in enumerate(shown[:2]):
                ui.blit(frame, icons.get(EMOJI[p], ui.u(54)), (w - ui.u(160) + k * ui.u(64), h - ui.u(60)))
        self.particles.update(dt)
        self.particles.draw(frame)
        self._draw_pops(frame)
        return None

    def _play(self, frame, poses, now, dt):
        h, w = frame.shape[:2]
        T = self.text
        st = now - self.state_t
        if self.state == "ready":
            T.draw(frame, "Prêt…", (w // 2, h // 2), 50, ui.WHITE, "mm", bold=True, stroke=3, alpha=0.7)
            if st > 0.9:
                self._new_cmd()
                self.state, self.state_t = "cmd", now
        elif self.state == "cmd":
            c = self.cmd
            if c["simon"]:
                T.draw(frame, "Jacques a dit :", (w // 2, ui.u(150)), 40, (80, 220, 255), "mt", bold=True, stroke=3)
            T.draw(frame, c["label"].upper() + " !", (w // 2, ui.u(215)), 56, ui.WHITE, "mt", bold=True, stroke=3)
            icon = "👏" if c["pose"] == "clap" else EMOJI[c["pose"]]
            ui.blit(frame, icons.get(icon, ui.u(170)), (w // 2, h // 2 + ui.u(40)))
            frac = max(0.0, 1 - st / c["window"])
            cv2.rectangle(frame, (int(w * (1 - frac) / 2), h - ui.u(24)), (int(w * (1 + frac) / 2), h - ui.u(12)),
                          ui.HUD_GREEN if frac > 0.3 else ui.RED, -1)
            if c["pose"] == "clap":
                doing = self.clap_t > c["t"]
                c["held"] = 1.0 if doing else 0.0
            else:
                doing = c["pose"] in poses
                c["held"] = c["held"] + dt if doing else 0.0
                if c["already"] and not doing:
                    c["already"] = False  # il faut refaire le geste, pas juste le garder
            done = c["held"] >= 0.25 and not (c["already"] and not c["simon"])
            if c["simon"] and done:
                self._win(now, st / c["window"])
            elif not c["simon"] and done:
                self._lose("Jacques n'a rien dit !")
            elif st >= c["window"]:
                if c["simon"]:
                    self._lose("Trop tard !")
                else:
                    self._win(now, 1.0, trap=True)
        elif self.state == "result":
            txt, col = self.result
            T.draw(frame, txt, (w // 2, h // 2), 64, col, "mm", bold=True, stroke=4)
            if st > 1.0:
                if self.lives <= 0:
                    self._finish()
                else:
                    self.state, self.state_t = "ready", now
        T.draw(frame, str(self.score), (ui.u(24), ui.u(22)), 44, ui.WHITE, "lt", bold=True, stroke=3)
        if self.streak >= 2:
            T.draw(frame, f"série x{self.streak}", (ui.u(24), ui.u(78)), 24, ui.HUD_GREEN, "lt", bold=True, stroke=2)
        for i in range(3):
            ui.blit(frame, icons.get("❤️" if i < self.lives else "🖤", ui.u(40)),
                    (w - ui.u(40) - (2 - i) * ui.u(52), ui.u(44)), 1.0 if i < self.lives else 0.5)

    def _win(self, now, frac, trap=False):
        self.streak += 1
        self.best_streak = max(self.best_streak, self.streak)
        pts = 50 if trap else 100 + int(150 * (1 - frac))
        pts *= 1 + min(3, self.streak // 5)
        self.score += pts
        self.result = ("BIEN VU !" if trap else "BRAVO !", ui.HUD_GREEN)
        self._popup(f"+{pts}", (self.w // 2, self.h // 2 + ui.u(90)), ui.WHITE, 40)
        self.particles.emit((self.w // 2, self.h // 2), 50, hue=0.33, speed=400)
        self._sfx(A.sfx_coin(), 0.5)
        self.state, self.state_t = "result", now

    def _lose(self, msg):
        self.lives -= 1
        self.streak = 0
        self.result = (msg, ui.RED)
        self._sfx(A.sfx_bad(), 0.5)
        self.state, self.state_t = "result", time.monotonic()


# =========================================================================== ESQUIVE
class DodgeGame(_Game):
    needs_hands = False
    needs_face = True

    def __init__(self, text, audio, records):
        super().__init__(text, audio, records, "esquive")
        self.rocks = []
        self.lives = 3
        self.inv_until = 0.0
        self.t_play = 0.0
        self.spawn_acc = 0.0
        self.stars = 0
        self.ship = None
        self.paused = False
        self.shake = 0.0

    def update(self, frame, tracker, dt):
        h, w = frame.shape[:2]
        self.w, self.h = w, h
        T = self.text
        now = time.monotonic()
        face = tracker.face()
        if face is not None:
            c = face.center
            r = max(ui.u(40), face.width * 0.45)
            self.ship = (c, r) if self.ship is None else (self.ship[0] * 0.4 + c * 0.6, r)
        e = now - self.t_phase
        if self.phase == "wait":
            ui.dim(frame, 0.6)
            T.draw(frame, "ESQUIVE", (w // 2, ui.u(60)), 64, ui.WHITE, "mt", bold=True, stroke=3)
            T.draw(frame, "Bouge la tête pour éviter les météores · attrape les étoiles · 3 vies",
                   (w // 2, ui.u(140)), 24, ui.WHITE, "mt", stroke=2)
            self.hold = self.hold + dt if face is not None else max(0.0, self.hold - dt)
            T.draw(frame, "Montre ton visage" if face is None else "C'est parti…", (w // 2, h - ui.u(140)), 30,
                   ui.HUD_GREEN if face is not None else ui.WHITE, "mm", bold=True, stroke=2)
            ui.arc(frame, (w // 2, h - ui.u(80)), ui.u(28), self.hold / 0.8, ui.GREEN, 5)
            if self.hold >= 0.8:
                self._set("countdown")
        elif self.phase == "countdown":
            left = 3 - e
            ui.dim(frame, 0.8)
            if left <= 0:
                self._set("play")
                self.start_music(2, 0.3)
            else:
                T.draw(frame, str(int(left) + 1), (w // 2, h // 2), 140, ui.WHITE, "mm", bold=True, stroke=4,
                       alpha=0.4 + 0.6 * (left - int(left)))
        elif self.phase == "play":
            self._play(frame, face, dt, now)
        if self.phase == "end":
            self._end_screen(frame, [("Score", str(self.score)), ("Temps", f"{self.t_play:.1f} s"),
                                     ("Étoiles", str(self.stars)), ("Record", str(self.records.get("esquive", 0)))])
        if self.ship is not None and self.phase in ("play", "countdown"):
            c, r = self.ship
            blink = now < self.inv_until and int(now * 12) % 2 == 0
            if not blink:
                cv2.circle(frame, (int(c[0]), int(c[1])), int(r), (255, 230, 120), ui.u(3), cv2.LINE_AA)
        self.particles.update(dt)
        self.particles.draw(frame)
        self._draw_pops(frame)
        return None

    def _play(self, frame, face, dt, now):
        h, w = frame.shape[:2]
        T = self.text
        if face is None:
            T.draw(frame, "PAUSE — montre ton visage", (w // 2, h // 2), 44, ui.WHITE, "mm", bold=True, stroke=3)
            return
        self.t_play += dt
        tp = self.t_play
        self.score = int(tp * 10) + self.stars * 50
        self.spawn_acc += (1.0 + tp * 0.07) * dt
        while self.spawn_acc >= 1:
            self.spawn_acc -= 1
            star = random.random() < 0.15
            side = tp > 20 and random.random() < 0.3
            r = ui.u(random.uniform(30, 55) if not star else 34)
            if side:
                x, y = (-r if random.random() < 0.5 else w + r), random.uniform(0.1, 0.6) * h
                vx = ui.u(random.uniform(220, 340)) * (1 if x < 0 else -1)
                vy = ui.u(random.uniform(80, 180))
            else:
                x, y = random.uniform(0.05, 0.95) * w, -r
                vx = ui.u(random.uniform(-60, 60))
                vy = ui.u(random.uniform(260, 420) + tp * 6)
            self.rocks.append({"x": x, "y": y, "vx": vx, "vy": vy, "r": r, "star": star,
                               "ang": random.uniform(0, 360), "spin": random.uniform(-200, 200)})
        c, sr = self.ship
        for rk in self.rocks:
            rk["x"] += rk["vx"] * dt
            rk["y"] += rk["vy"] * dt
            rk["ang"] += rk["spin"] * dt
            if not rk["star"] and random.random() < 0.6:
                self.particles.emit((rk["x"] - rk["vx"] * 0.03, rk["y"] - rk["vy"] * 0.03), 1, hue=0.07, speed=40)
            if math.hypot(rk["x"] - c[0], rk["y"] - c[1]) < rk["r"] * 0.8 + sr:
                if rk["star"]:
                    rk["dead"] = True
                    self.stars += 1
                    self._popup("+50", (rk["x"], rk["y"]), (60, 220, 255))
                    self.particles.emit((rk["x"], rk["y"]), 40, hue=0.14, speed=350)
                    self._sfx(A.sfx_coin(), 0.5)
                elif now > self.inv_until:
                    rk["dead"] = True
                    self.lives -= 1
                    self.inv_until = now + 1.5
                    self.shake = 1.0
                    self.particles.emit((rk["x"], rk["y"]), 90, hue=0.05, speed=600)
                    self._sfx(A.sfx_boom(), 0.7)
        self.rocks = [rk for rk in self.rocks if not rk.get("dead") and -ui.u(200) < rk["x"] < w + ui.u(200)
                      and rk["y"] < h + ui.u(120)]
        for rk in self.rocks:
            spr = icons.get("⭐" if rk["star"] else "🪨", rk["r"] * 2)
            ui.blit(frame, _rot_sprite(spr, rk["ang"]) if not rk["star"] else spr, (rk["x"], rk["y"]))
        if self.shake > 0:
            self.shake = max(0.0, self.shake - dt * 2.5)
            frame[:] = np.roll(frame, int(math.sin(now * 70) * ui.u(20) * self.shake), axis=1)
        T.draw(frame, str(self.score), (ui.u(24), ui.u(22)), 44, ui.WHITE, "lt", bold=True, stroke=3)
        T.draw(frame, f"{tp:.1f} s", (ui.u(24), ui.u(80)), 22, ui.WHITE, "lt", stroke=2, alpha=0.8)
        for i in range(3):
            ui.blit(frame, icons.get("❤️" if i < self.lives else "🖤", ui.u(40)),
                    (w - ui.u(40) - (2 - i) * ui.u(52), ui.u(44)), 1.0 if i < self.lives else 0.5)
        if self.lives <= 0:
            self._finish()


# =========================================================================== ARCADE
GAMES = [("danse", "Danse", "icon_danse", "Reproduis les poses en rythme", DanceGame),
         ("bulles", "Bulles", "icon_bulles", "Éclate les bulles, attrape les bonus", BubbleGame),
         ("tranche", "Tranche", "icon_tranche", "Coupe les fruits d'un geste vif", SliceGame),
         ("jacques", "Jacques a dit", "icon_jacques", "Le bon geste… si Jacques l'a dit", SimonGame),
         ("esquive", "Esquive", "icon_esquive", "Évite les météores avec ta tête", DodgeGame)]


class Arcade:
    needs_hands = True
    needs_face = needs_seg = needs_pose = False

    def __init__(self, text, audio, records):
        self.text, self.audio, self.records = text, audio, records
        self.sel = 0
        self.hover = None
        self.hover_t = 0.0
        self.t0 = time.monotonic()
        self.particles = ui.Particles()

    def key(self, k):
        if k in (ESC, ord("q")):
            return "exit"
        if ord("1") <= k <= ord("5"):
            return ("launch", k - ord("1"))
        if k in (ord("n"), 3):
            self.sel = (self.sel + 1) % len(GAMES)
        elif k in (ord("b"), 2):
            self.sel = (self.sel - 1) % len(GAMES)
        elif k in (ENTER, SPACE):
            return ("launch", self.sel)
        return None

    def update(self, frame, tracker, dt):
        h, w = frame.shape[:2]
        T = self.text
        now = time.monotonic()
        frame[:] = cv2.convertScaleAbs(ui.soft_blur(frame, 16 * ui.S), alpha=0.45)
        T.draw(frame, "ARCADE", (w // 2, ui.u(50)), 64, ui.WHITE, "mt", bold=True, stroke=3)
        T.draw(frame, "Pointe un jeu avec ton index · ou touches 1-5 · ÉCHAP pour revenir aux effets",
                   (w // 2, ui.u(130)), 20, ui.WHITE, "mt", alpha=0.8)
        cw, ch = ui.u(205), ui.u(300)
        gap = ui.u(22)
        x0 = w // 2 - (len(GAMES) * cw + (len(GAMES) - 1) * gap) // 2
        y0 = h // 2 - ch // 2 + ui.u(30)
        tips = [hd.pts[8] for hd in tracker.hands()]
        hovered = None
        rects = []
        for i, (key, name, icon, desc, _) in enumerate(GAMES):
            x = x0 + i * (cw + gap)
            rects.append((x, y0, x + cw, y0 + ch))
            for p in tips:
                if x < p[0] < x + cw and y0 < p[1] < y0 + ch:
                    hovered = i
        if hovered is not None:
            self.sel = hovered
            if self.hover != hovered:
                self.hover, self.hover_t = hovered, now
                if self.audio is not None:
                    self.audio.play(A.sfx_pop(1.2), 0.3)
        else:
            self.hover = None
        for i, (key, name, icon, desc, _) in enumerate(GAMES):
            x, y, x1, y1 = rects[i]
            cur = i == self.sel
            lift = ui.u(14) if cur else 0
            ui.panel(frame, x, y - lift, x1, y1 - lift, 0.65 if cur else 0.45, 22)
            if cur:
                cv2.rectangle(frame, (x + ui.u(20), y1 - lift - ui.u(6)), (x1 - ui.u(20), y1 - lift - ui.u(2)),
                              ui.HUD_GREEN, -1)
            bob = math.sin(now * 3 + i) * ui.u(6) if cur else 0
            ui.blit(frame, icons.get(icon, ui.u(130)), ((x + x1) / 2, y + ui.u(90) - lift + bob))
            T.draw(frame, f"{i + 1}. {name}", ((x + x1) / 2, y + ui.u(170) - lift), 24, ui.WHITE, "mt", bold=True)
            T.draw(frame, desc, ((x + x1) / 2, y + ui.u(208) - lift), 14, ui.WHITE, "mt", alpha=0.75)
            rec = self.records.get(key, 0)
            T.draw(frame, f"record : {rec}" if rec else "pas encore de record", ((x + x1) / 2, y + ui.u(250) - lift),
                   15, ui.HUD_GREEN if rec else ui.WHITE, "mt", alpha=0.85)
            if cur and self.hover == i:
                ui.arc(frame, ((x + x1) / 2, y + ui.u(90) - lift), ui.u(72), (now - self.hover_t) / 0.8, ui.GREEN, 5)
        for p in tips:
            cv2.circle(frame, (int(p[0]), int(p[1])), ui.u(10), ui.WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(p[0]), int(p[1])), ui.u(16), ui.WHITE, ui.u(2), cv2.LINE_AA)
        self.particles.update(dt)
        self.particles.draw(frame)
        if self.hover is not None and now - self.hover_t >= 0.8:
            self.hover = None
            if self.audio is not None:
                self.audio.play(A.sfx_chime(), 0.4)
            return ("launch", self.sel)
        return None
