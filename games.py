"""Mini-jeux.

DanceGame  — façon Just Dance : des pictogrammes défilent en rythme, reproduis la pose avec tes bras
             pile sur le temps (suivi du corps MediaPipe). PARFAIT / SUPER / BIEN / RATÉ, combo, étoiles.
BubbleGame — éclate les bulles avec le bout de tes index, attrape les étoiles, évite les bombes.
"""
import math
import random
import threading
import time

import cv2
import numpy as np

import audio as A
import icons
import ui

ESC, SPACE = 27, 32
_TRACKS = {}
_SPOTS = {}
_TRACK_LOCK = threading.Lock()


def get_track(i):
    """Morceau i de audio.SONGS (généré une fois puis gardé en mémoire)."""
    with _TRACK_LOCK:
        if i not in _TRACKS:
            s = A.SONGS[i]
            _TRACKS[i] = A.make_track(s["bpm"], 26, s["seed"], s["prog"], s["style"])
        return _TRACKS[i]


def rank_letter(acc):
    for th, letter in ((0.9, "S"), (0.75, "A"), (0.6, "B"), (0.4, "C")):
        if acc >= th:
            return letter
    return "D"

# --------------------------------------------------------------------------- poses de danse
# Angles des segments (degrés, repère écran : 0 = vers la droite, 90 = vers le bas, -90 = vers le haut)
# (bras gauche de l'écran : (épaule→coude, coude→poignet)), (bras droit de l'écran : idem)
MOVES = {
    "bras en l'air": ((-100, -95), (-80, -85)),
    "T": ((180, 180), (0, 0)),
    "V": ((-135, -135), (-45, -45)),
    "disco droite": ((130, 130), (-45, -45)),
    "disco gauche": ((-135, -135), (50, 50)),
    "muscles": ((180, -90), (0, -90)),
    "M": ((-145, -35), (-35, -145)),
    "main droite": ((100, 95), (-90, -90)),
    "main gauche": ((-90, -90), (80, 85)),
    "hanches": ((120, 40), (60, 140)),
    "pointe à droite": ((100, 95), (-5, -5)),
    "pointe à gauche": ((185, 185), (80, 85)),
}
SECTIONS = [["T", "V", "T", "V"], ["disco droite", "disco gauche", "disco droite", "disco gauche"],
            ["main droite", "main gauche", "V", "bras en l'air"], ["muscles", "hanches", "muscles", "hanches"],
            ["pointe à droite", "pointe à gauche", "M", "T"], ["V", "M", "disco droite", "bras en l'air"]]
RATINGS = [(0.80, "PARFAIT", (90, 255, 140), 100), (0.64, "SUPER", (255, 220, 80), 75),
           (0.48, "BIEN", (60, 200, 255), 50), (0.0, "RATÉ", (90, 90, 255), 0)]


def _angdiff(a, b):
    return abs((a - b + 180) % 360 - 180)


def body_arms(body):
    """Renvoie ((épaule, coude, poignet) bras gauche écran, idem droit, visibilités) ou None."""
    if body is None:
        return None
    p, v = body.pts, body.vis
    a, b = (11, 13, 15), (12, 14, 16)
    if p[11, 0] > p[12, 0]:
        a, b = b, a
    return (p[list(a)], v[list(a)]), (p[list(b)], v[list(b)])


def _dir(deg):
    return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))], np.float32)


def _ang(v):
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


def clap_score(body):
    """Mains jointes devant soi (poignets proches)."""
    if body is None or min(body.vis[15], body.vis[16]) < 0.3:
        return 0.0
    sw = float(np.linalg.norm(body.pts[11] - body.pts[12])) + 1e-6
    d = float(np.linalg.norm(body.pts[15] - body.pts[16])) / sw
    return float(np.clip(1 - (d - 0.35) / 0.5, 0, 1))


def pose_score(body, move):
    """Note 0..1. Moitié : direction du bras entier (épaule→poignet, très stable),
    moitié : angle de chaque segment (pour les poses où le coude est plié)."""
    if move == "CLAP":
        return clap_score(body)
    arms = body_arms(body)
    if arms is None:
        return 0.0
    seg_total = whole_total = 0.0
    for (pts, vis), (up, fore) in zip(arms, MOVES[move]):
        for i, target in ((0, up), (1, fore)):
            if vis[i] >= 0.35 and vis[i + 1] >= 0.35:
                seg_total += float(np.clip(1 - (_angdiff(_ang(pts[i + 1] - pts[i]), target) - 10) / 35, 0, 1))
        if vis[0] >= 0.35 and vis[2] >= 0.35:
            target = _ang(_dir(up) * 1.1 + _dir(fore))  # direction épaule→poignet de la pose cible
            whole_total += float(np.clip(1 - (_angdiff(_ang(pts[2] - pts[0]), target) - 10) / 30, 0, 1))
    return 0.5 * seg_total / 4 + 0.5 * whole_total / 2


def framing_hint(body, w, h):
    """Placement pour la danse : (ok, message). Il faut assez de recul pour que les mains levées restent
    dans l'image."""
    if body is None:
        return False, "Mets-toi face à la caméra"
    p, v = body.pts, body.vis
    if min(v[11], v[12]) < 0.5:
        return False, "On ne voit pas tes épaules"
    sw = float(np.linalg.norm(p[11] - p[12])) / w
    cx = float(p[11, 0] + p[12, 0]) / 2 / w
    sy = float(p[11, 1] + p[12, 1]) / 2 / h
    if sw > 0.24:
        return False, "Recule d'un pas (tes mains sortiraient de l'image)"
    if sw < 0.09:
        return False, "Rapproche-toi un peu"
    if not 0.3 < cx < 0.7:
        return False, "Mets-toi au centre"
    if sy < 0.32:
        return False, "Baisse un peu la caméra (ou recule)"
    if sy > 0.72:
        return False, "Monte un peu la caméra (ou recule)"
    if min(v[13], v[14], v[15], v[16]) < 0.4:
        return False, "Écarte les bras pour qu'on voie tes coudes et tes mains"
    return True, "Parfait, ne bouge plus…"


def draw_picto(img, center, size, move, color=(255, 255, 255), alpha=1.0, thickness=None):
    """Bonhomme bâton qui montre la pose."""
    cx, cy = center
    s = size
    th = thickness or max(2, int(s * 0.07))
    over = img.copy() if alpha < 1 else img
    head = (int(cx), int(cy - s * 0.32))
    sh_l = np.array([cx - s * 0.16, cy - s * 0.16])
    sh_r = np.array([cx + s * 0.16, cy - s * 0.16])
    hip = (int(cx), int(cy + s * 0.22))
    cv2.circle(over, head, int(s * 0.1), color, -1, cv2.LINE_AA)
    cv2.line(over, (int(cx), int(cy - s * 0.18)), hip, color, th, cv2.LINE_AA)
    cv2.line(over, tuple(sh_l.astype(int)), tuple(sh_r.astype(int)), color, th, cv2.LINE_AA)
    for leg in (-1, 1):
        cv2.line(over, hip, (int(cx + leg * s * 0.12), int(cy + s * 0.48)), color, th, cv2.LINE_AA)
    if move == "CLAP":
        hands = np.array([cx, cy - s * 0.05])
        for sh in (sh_l, sh_r):
            e = sh + np.array([np.sign(sh[0] - cx) * s * 0.12, s * 0.16])
            cv2.line(over, tuple(sh.astype(int)), tuple(e.astype(int)), color, th, cv2.LINE_AA)
            cv2.line(over, tuple(e.astype(int)), tuple(hands.astype(int)), color, th, cv2.LINE_AA)
        if alpha < 1:
            cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)
        ui.blit(img, icons.get("👏", int(s * 0.32)), (cx + s * 0.3, cy - s * 0.3), alpha)
        return
    for sh, (up, fore) in zip((sh_l, sh_r), MOVES[move]):
        e = sh + s * 0.2 * np.array([math.cos(math.radians(up)), math.sin(math.radians(up))])
        wr = e + s * 0.18 * np.array([math.cos(math.radians(fore)), math.sin(math.radians(fore))])
        cv2.line(over, tuple(sh.astype(int)), tuple(e.astype(int)), color, th, cv2.LINE_AA)
        cv2.line(over, tuple(e.astype(int)), tuple(wr.astype(int)), color, th, cv2.LINE_AA)
    if alpha < 1:
        cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)


def draw_star(img, center, r, fill, color=(60, 220, 255)):
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.45
        pts.append((center[0] + math.cos(a) * rr, center[1] + math.sin(a) * rr))
    pts = np.array(pts, np.int32)
    cv2.polylines(img, [pts], True, (200, 200, 200), max(1, ui.u(2)), cv2.LINE_AA)
    if fill > 0:
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(m, [pts], 255, cv2.LINE_AA)
        x0 = int(center[0] - r)
        m[:, int(x0 + 2 * r * fill):] = 0
        img[m > 0] = color


class _Game:
    """Base commune : écran d'attente, compte à rebours, résultats."""
    needs_face = needs_seg = False
    needs_hands = True
    needs_pose = False

    def __init__(self, text, audio, records, key):
        self.text, self.audio, self.records, self.rkey = text, audio, records, key
        self.particles = ui.Particles(1500)
        self.phase = "wait"
        self.t_phase = time.monotonic()
        self.hold = 0.0
        self.score = 0
        self.new_record = False
        self.pops = []  # (texte, x, y, t0, couleur, taille)
        self.music = None
        self.w = self.h = 0

    def _set(self, ph):
        self.phase, self.t_phase = ph, time.monotonic()

    def key(self, k):
        if k in (ESC, ord("q")):
            self.stop_music()
            return "exit"
        if k == SPACE and self.phase in ("end", "wait"):
            self._restart() if self.phase == "end" else self._set("countdown")
        return None

    def _restart(self):
        self.stop_music()
        self.__init__(self.text, self.audio, self.records)

    def start_music(self, song=0, vol=0.35):
        """Musique de fond en boucle (préparée dans un thread pour ne pas figer l'image)."""
        def run():
            data, _ = get_track(song)
            if self.audio is not None and self.phase == "play" and self.music is None:
                self.music = self.audio.play(data, vol, loop=True)
        threading.Thread(target=run, daemon=True).start()

    def stop_music(self):
        if self.audio is not None and self.music is not None:
            self.audio.stop(self.music)
        self.music = None

    def _sfx(self, data, vol=0.8):
        if self.audio is not None:
            self.audio.play(data, vol)

    def _popup(self, txt, pos, color, size=44):
        self.pops.append((txt, float(pos[0]), float(pos[1]), time.monotonic(), color, size))

    def _draw_pops(self, frame):
        now = time.monotonic()
        keep = []
        for txt, x, y, t0, col, size in self.pops:
            e = (now - t0) / 0.9
            if e >= 1:
                continue
            keep.append((txt, x, y, t0, col, size))
            sc = 1 + 0.3 * math.sin(min(1, e * 4) * math.pi) if e < 0.25 else 1
            self.text.draw(frame, txt, (x, y - e * ui.u(60)), size * sc, col, "mm", bold=True, stroke=3,
                           alpha=1 - ui.ease_in_out((e - 0.6) / 0.4) if e > 0.6 else 1)
        self.pops = keep

    def _finish(self):
        self.stop_music()
        best = self.records.get(self.rkey, 0)
        self.new_record = self.score > best
        if self.new_record:
            self.records[self.rkey] = self.score
            for k in range(6):
                self.particles.emit((random.uniform(0.2, 0.8) * self.w, self.h * 0.4), 60, hue=k / 6, speed=420)
        self._sfx(A.sfx_chime(), 0.7)
        self._set("end")

    def _end_screen(self, frame, lines, stars=None):
        h, w = frame.shape[:2]
        T = self.text
        ui.dim(frame, 0.45)
        gap = ui.u(60) if stars is not None else 0
        pw, ph = ui.u(620), ui.u(140 + 44 * len(lines)) + gap
        x0, y0 = (w - pw) // 2, (h - ph) // 2
        ui.panel(frame, x0, y0, x0 + pw, y0 + ph, 0.7, 24)
        T.draw(frame, "Terminé !", (w // 2, y0 + ui.u(26)), 40, ui.WHITE, "mt", bold=True)
        if stars is not None:
            for i in range(5):
                draw_star(frame, (w // 2 + (i - 2) * ui.u(56), y0 + ui.u(108)), ui.u(22), 1.0 if i < stars else 0.0)
        y = y0 + ui.u(92) + gap
        for k, v in lines:
            T.draw(frame, k, (x0 + ui.u(40), y), 22, ui.WHITE, "lt", alpha=0.8)
            T.draw(frame, v, (x0 + pw - ui.u(40), y), 22, ui.WHITE, "rt", bold=True)
            y += ui.u(44)
        if self.new_record:
            T.draw(frame, "NOUVEAU RECORD !", (w // 2, y + ui.u(4)), 24, ui.HUD_GREEN, "mt", bold=True)
        T.draw(frame, "ESPACE rejouer   ·   ÉCHAP retour aux effets", (w // 2, y0 + ph + ui.u(26)), 18, ui.WHITE,
               "mt", stroke=2, alpha=0.8)


# =========================================================================== DANSE
class DanceGame(_Game):
    needs_pose = True
    needs_hands = False
    LEAD = 2.2  # secondes d'avance avec lesquelles un pictogramme apparaît

    song_i = 0  # gardé entre deux parties

    def __init__(self, text, audio, records):
        super().__init__(text, audio, records, "danse")
        self.track = None
        self.bpm = 118
        self.voice = None
        self.t0 = None
        self.moves = []
        self._prep_for = None
        self._raise = {"l": 0.0, "r": 0.0}
        self.last_song_switch = 0.0
        self._prepare_async()
        self.combo = self.best_combo = 0
        self.results = []
        self.last_beat = -1
        self.beat_flash = 0.0
        self.w = self.h = 0
        self.feedback = None

    def _prepare_async(self):
        i = DanceGame.song_i
        self._prep_for = i
        self.track = None

        def run():
            track, bpm = get_track(i)
            if self._prep_for != i:
                return
            beat = 60 / bpm
            rng = random.Random(A.SONGS[i]["seed"])
            secs = SECTIONS[:]
            rng.shuffle(secs)
            seq = []
            for sec in secs:
                seq += sec + sec  # chaque section : 4 poses x 2 (on apprend puis on enchaîne)
            hits = [8 + 2 * k for k in range((26 * 4 - 12) // 2)]
            moves = []
            for k, b in enumerate(hits):
                name = "CLAP" if k % 12 == 11 else seq[k % len(seq)]
                moves.append({"name": name, "t": b * beat, "best": 0.0, "done": False, "gold": k % 8 == 7})
            self.moves, self.bpm, self.track = moves, bpm, track
        threading.Thread(target=run, daemon=True).start()

    def _switch_song(self, step):
        DanceGame.song_i = (DanceGame.song_i + step) % len(A.SONGS)
        self.last_song_switch = time.monotonic()
        self._sfx(A.sfx_whoosh(), 0.4)
        self._prepare_async()

    def _song_cards(self, frame):
        h, w = frame.shape[:2]
        T = self.text
        n = len(A.SONGS)
        for k, s in enumerate(A.SONGS):
            x = w // 2 + (k - (n - 1) / 2) * ui.u(250)
            y = ui.u(215)
            cur = k == DanceGame.song_i
            ui.panel(frame, x - ui.u(110), y - ui.u(32), x + ui.u(110), y + ui.u(32), 0.6 if cur else 0.35, 16)
            if cur:
                cv2.rectangle(frame, (int(x - ui.u(110)), int(y + ui.u(30))), (int(x + ui.u(110)), int(y + ui.u(34))),
                              ui.HUD_GREEN, -1)
            T.draw(frame, s["name"], (x, y - ui.u(8)), 20 if cur else 17, ui.WHITE, "mm", bold=cur,
                   alpha=1 if cur else 0.6)
            T.draw(frame, f"{s['bpm']} BPM", (x, y + ui.u(15)), 14, ui.WHITE, "mm", alpha=0.6)
        T.draw(frame, "Lève une main au-dessus de la tête (ou 1 / 2 / 3) pour changer de musique",
               (w // 2, ui.u(268)), 16, ui.WHITE, "mt", stroke=2, alpha=0.75)

    def _raise_hand(self, body, dt):
        """Main levée au-dessus de la tête 0,5 s : musique précédente (gauche) / suivante (droite)."""
        if body is None or time.monotonic() - self.last_song_switch < 1.2:
            return
        arms = body_arms(body)
        nose_y = body.pts[0, 1]
        for side, (pts, vis) in zip(("l", "r"), arms):
            up = vis[2] > 0.5 and pts[2, 1] < nose_y - 0.4 * float(np.linalg.norm(body.pts[11] - body.pts[12]))
            self._raise[side] = self._raise[side] + dt if up else 0.0
            if self._raise[side] > 0.5:
                self._raise = {"l": 0.0, "r": 0.0}
                self._switch_song(-1 if side == "l" else 1)
                return

    def _restart(self):
        if self.audio is not None:
            self.audio.stop(self.voice)
        self.__init__(self.text, self.audio, self.records)

    def key(self, k):
        if self.phase == "wait" and k in (ord("1"), ord("2"), ord("3")):
            DanceGame.song_i = k - ord("1")
            self._prepare_async()
            return None
        if self.phase == "wait" and k in (ord("n"), ord("b")):
            self._switch_song(1 if k == ord("n") else -1)
            return None
        r = super().key(k)
        if r == "exit" and self.audio is not None:
            self.audio.stop(self.voice)
        return r

    def _clock(self):
        if self.voice is not None and self.audio is not None and self.audio.ok:
            return self.audio.voice_time(self.voice)
        return time.monotonic() - self.t0

    def update(self, frame, tracker, dt):
        h, w = frame.shape[:2]
        self.w, self.h = w, h
        T = self.text
        now = time.monotonic()
        body = tracker.body()

        ui.dim(frame, 0.6)
        if self.phase == "play":
            self._lights(frame)
        self._draw_body(frame, body)

        if self.phase == "wait":
            self._song_cards(frame)
            self._raise_hand(body, dt)
            T.draw(frame, "DANSE", (w // 2, ui.u(60)), 64, ui.WHITE, "mt", bold=True, stroke=3)
            T.draw(frame, "Reproduis les poses avec tes bras, pile sur le temps · poses dorées = x2 · CLAP = mains jointes",
                   (w // 2, ui.u(140)), 22, ui.WHITE, "mt", stroke=2)
            ok, msg = framing_hint(body, w, h)
            self.hold = self.hold + dt if ok else max(0.0, self.hold - dt)
            # silhouette guide : là où tes épaules doivent être, bras en V
            draw_picto(frame, (w // 2, int(h * 0.58)), int(h * 0.62), "V", ui.HUD_GREEN if ok else ui.WHITE,
                       0.25 if not ok else 0.4, thickness=ui.u(5))
            T.draw(frame, msg, (w // 2, h - ui.u(120)), 26, ui.HUD_GREEN if ok else ui.WHITE, "mm", bold=True,
                   stroke=2)
            ui.arc(frame, (w // 2, h - ui.u(180)), ui.u(26), self.hold / 1.2, ui.GREEN, 5)
            T.draw(frame, "ESPACE pour démarrer quand même · ÉCHAP retour", (w // 2, h - ui.u(40)), 18, ui.WHITE,
                   "mm", alpha=0.6)
            if self.hold >= 1.2 and self.track is not None:
                self._set("countdown")
        elif self.phase == "countdown":
            if self.track is None:
                T.draw(frame, "Préparation de la musique…", (w // 2, h // 2), 30, ui.WHITE, "mm", bold=True)
                return None
            if self.t0 is None:
                self.t0 = now
                if self.audio is not None and self.audio.ok:
                    self.voice = self.audio.play(self.track, 0.8)
                self._set("play")
        if self.phase == "play":
            self._play(frame, body, dt)
        elif self.phase == "end":
            n = max(1, len(self.results))
            acc = sum(r[3] for r in self.results) / (100 * n)
            stars = sum(acc >= th for th in (0.2, 0.4, 0.55, 0.7, 0.85))
            counts = {name: sum(1 for r in self.results if r[1] == name) for _, name, _, _ in RATINGS}
            self._end_screen(frame, [("Musique", A.SONGS[DanceGame.song_i]["name"]),
                                     ("Note", rank_letter(acc)),
                                     ("Score", f"{self.score:,}".replace(",", " ")),
                                     ("Précision", f"{acc * 100:.0f} %"),
                                     ("Meilleur combo", str(self.best_combo)),
                                     ("Parfait / Super / Bien / Raté",
                                      f"{counts['PARFAIT']} / {counts['SUPER']} / {counts['BIEN']} / {counts['RATÉ']}"),
                                     ("Record", f"{self.records.get('danse', 0):,}".replace(",", " "))], stars)
        self.particles.update(dt)
        self.particles.draw(frame)
        self._draw_pops(frame)
        return None

    def _draw_body(self, frame, body):
        if body is None:
            return
        self._arm_lengths(body)
        p, v = body.pts / 2, body.vis  # calque néon en demi-résolution (2x plus rapide, identique à l'œil)
        h, w = frame.shape[:2]
        base = np.array(ui.hsv_color(time.monotonic() * 0.1, 0.5, 1.0), np.float32)
        layer = np.zeros((h // 2, w // 2, 3), np.uint8)
        for a, b in ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)):
            q = min(v[a], v[b])
            if q > 0.25:
                col = tuple(int(c) for c in base * min(1.0, q * 1.4))
                cv2.line(layer, tuple(p[a].astype(int)), tuple(p[b].astype(int)), col, max(2, ui.u(3.5)), cv2.LINE_AA)
        for i in (15, 16):
            if v[i] > 0.25:
                cv2.circle(layer, tuple(p[i].astype(int)), ui.u(6), tuple(int(c) for c in base), -1, cv2.LINE_AA)
        layer = cv2.addWeighted(layer, 0.8, ui.soft_blur(layer, ui.u(5)), 1.6, 0)
        cv2.add(frame, cv2.resize(layer, (w, h), interpolation=cv2.INTER_LINEAR), frame)

    def _arm_lengths(self, body):
        """Longueurs des bras lissées (sinon le fantôme tremble avec le coude)."""
        arms = body_arms(body)
        if arms is None:
            return
        if not hasattr(self, "arm_len"):
            sw = float(np.linalg.norm(body.pts[11] - body.pts[12]))
            self.arm_len = [[0.8 * sw, 0.75 * sw], [0.8 * sw, 0.75 * sw]]
        for k, (pts, vis) in enumerate(arms):
            for i in (0, 1):
                if vis[i] > 0.6 and vis[i + 1] > 0.6:
                    L = float(np.linalg.norm(pts[i + 1] - pts[i]))
                    self.arm_len[k][i] = 0.92 * self.arm_len[k][i] + 0.08 * L

    def _lights(self, frame):
        """Projecteurs de couleur qui balaient la scène et pulsent sur le temps."""
        h, w = frame.shape[:2]
        key = ("spot", w, h)
        if key not in _SPOTS:
            sw, sh = w // 8, h // 8
            yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
            _SPOTS[key] = (xx, yy)
        xx, yy = _SPOTS[key]
        t = self._clock()
        layer = np.zeros(xx.shape + (3,), np.float32)
        for k in range(3):
            cx = (0.5 + 0.4 * math.sin(t * (0.5 + 0.2 * k) + k * 2)) * xx.shape[1]
            cy = (0.25 + 0.15 * math.cos(t * 0.7 + k)) * xx.shape[0]
            g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (xx.shape[1] * 0.18) ** 2))
            layer += g[..., None] * np.array(ui.hsv_color(t * 0.05 + k / 3, 0.8, 1.0), np.float32)
        layer *= 0.35 + 0.45 * self.beat_flash
        cv2.add(frame, cv2.resize(np.clip(layer, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR),
                frame)

    def _ghost(self, frame, body, move, alpha):
        arms = body_arms(body)
        if move == "CLAP":
            if body is not None:
                c = (body.pts[11] + body.pts[12]) / 2 + (0, float(np.linalg.norm(body.pts[11] - body.pts[12])) * 0.6)
                ui.blit(frame, icons.get("👏", ui.u(90)), c, alpha * 2)
            return
        if arms is None or not hasattr(self, "arm_len"):
            return
        over = frame.copy()
        for k, ((pts, vis), (up, fore)) in enumerate(zip(arms, MOVES[move])):
            sh = pts[0]
            l1, l2 = self.arm_len[k]
            e = sh + l1 * np.array([math.cos(math.radians(up)), math.sin(math.radians(up))])
            wr = e + l2 * np.array([math.cos(math.radians(fore)), math.sin(math.radians(fore))])
            for a, b in ((sh, e), (e, wr)):
                cv2.line(over, tuple(a.astype(int)), tuple(b.astype(int)), (255, 240, 120), ui.u(16), cv2.LINE_AA)
        cv2.addWeighted(over, alpha, frame, 1 - alpha, 0, frame)

    def _play(self, frame, body, dt):
        h, w = frame.shape[:2]
        T = self.text
        t = self._clock()
        beat_len = 60 / self.bpm
        beat = int(t / beat_len) if t >= 0 else -1
        if beat != self.last_beat:
            self.last_beat = beat
            self.beat_flash = 1.0
        self.beat_flash = max(0.0, self.beat_flash - dt * 4)

        # compte à rebours sur l'intro (temps 4 à 7)
        if 4 <= beat < 8:
            txt = ["3", "2", "1", "GO !"][beat - 4]
            T.draw(frame, txt, (w // 2, h // 2 - ui.u(40)), 130, ui.WHITE, "mm", bold=True, stroke=4,
                   alpha=0.4 + 0.6 * self.beat_flash)
        elif beat < 4:
            T.draw(frame, "Prépare-toi…", (w // 2, h // 2 - ui.u(40)), 50, ui.WHITE, "mm", bold=True, stroke=3)

        # jugement
        upcoming = None
        for m in self.moves:
            if m["done"]:
                continue
            if t >= m["t"] - 0.4:
                m["best"] = max(m["best"], pose_score(body, m["name"]))
            if t > m["t"] + 0.3:
                m["done"] = True
                self._judge(m, body)
            elif upcoming is None:
                upcoming = m
        if upcoming is not None and upcoming["t"] - t < 1.2:
            self._ghost(frame, body, upcoming["name"], 0.35 * ui.ease_out(1 - (upcoming["t"] - t) / 1.2))

        ok, msg = framing_hint(body, w, h)
        if not ok and beat >= 4:
            T.draw(frame, msg, (w // 2, h - ui.u(215)), 20, (80, 200, 255), "mm", bold=True, stroke=2)

        # bande des pictogrammes
        lane_y = h - ui.u(110)
        judge_x = int(w * 0.18)
        ui.panel(frame, 0, lane_y - ui.u(85), w, lane_y + ui.u(85), 0.45, 0)
        pulse = int(ui.u(8) * self.beat_flash)
        cv2.circle(frame, (judge_x, lane_y), ui.u(74) + pulse, ui.HUD_GREEN, ui.u(4), cv2.LINE_AA)
        speed = (w - judge_x) / self.LEAD
        for m in self.moves:
            dtm = m["t"] - t
            if -0.5 < dtm < self.LEAD + 0.3:
                x = judge_x + dtm * speed
                near = max(0.0, 1 - abs(dtm) / 0.6)
                size = ui.u(120) * (1 + 0.25 * near)
                a = 0.35 if m["done"] else 1.0
                col = (60, 210, 255) if m["gold"] else ui.WHITE
                col = col if not m["done"] else (150, 150, 150)
                if m["gold"] and not m["done"]:
                    cv2.circle(frame, (int(x), lane_y), int(size * 0.55), (60, 210, 255), ui.u(3), cv2.LINE_AA)
                draw_picto(frame, (x, lane_y), size, m["name"], col, a)
        if upcoming is not None and upcoming["t"] - t < self.LEAD:
            label = "APPLAUDIS !" if upcoming["name"] == "CLAP" else upcoming["name"].upper()
            if upcoming["gold"]:
                label = label + "  ·  DORÉE x2"
            T.draw(frame, label, (w // 2, ui.u(70)), 44, (60, 210, 255) if upcoming["gold"] else ui.WHITE, "mt",
                   bold=True, stroke=3)
            ui.arc(frame, (w // 2, ui.u(160)), ui.u(26), 1 - max(0.0, upcoming["t"] - t) / self.LEAD, ui.HUD_GREEN, 5)

        # score et combo
        T.draw(frame, f"{self.score:,}".replace(",", " "), (ui.u(24), ui.u(18)), 40, ui.WHITE, "lt", bold=True,
               stroke=3)
        if self.combo >= 2:
            T.draw(frame, f"combo x{self.combo}", (ui.u(24), ui.u(72)), 24, ui.HUD_GREEN, "lt", bold=True, stroke=2)
        done = [m for m in self.moves if m["done"]]
        if done:
            acc = sum(r[3] for r in self.results) / (100 * len(self.moves))
            for i in range(5):
                draw_star(frame, (w - ui.u(40) - (4 - i) * ui.u(44), ui.u(40)), ui.u(17),
                          float(np.clip(acc / 0.85 * 5 - i, 0, 1)))
        if self.feedback is not None:
            txt, col, t0, pos = self.feedback
            e = time.monotonic() - t0
            if e < 0.8:
                sc = 1 + 0.4 * (1 - ui.ease_out(e / 0.25)) if e < 0.25 else 1
                T.draw(frame, txt, pos, 70 * sc, col, "mm", bold=True, stroke=4,
                       alpha=1 - ui.ease_in_out((e - 0.5) / 0.3) if e > 0.5 else 1)
        if self.moves and all(m["done"] for m in self.moves) and t > self.moves[-1]["t"] + 1.5:
            if self.audio is not None:
                self.audio.stop(self.voice)
            self._finish()

    def _judge(self, m, body):
        s = m["best"]
        for th, name, col, pts in RATINGS:
            if s >= th:
                break
        if pts:
            self.combo += 1
            self.best_combo = max(self.best_combo, self.combo)
        else:
            self.combo = 0
        mult = (1 + min(3, self.combo // 4)) * (2 if m["gold"] else 1)
        self.score += pts * mult
        self.results.append((m["name"], name, s, pts))
        pos = (self.w // 2, self.h // 2 - ui.u(60))
        if body is not None:
            pos = ((body.pts[11] + body.pts[12]) / 2 - (0, ui.u(140))).tolist()
        self.feedback = (name, col, time.monotonic(), pos)
        if pts >= 75:
            self.particles.emit(pos, 50 if pts == 100 else 25, hue=0.33 if pts == 100 else 0.13, speed=380)
            if m["gold"]:
                for k in range(5):
                    self.particles.emit(pos, 40, hue=0.12 + k * 0.01, speed=600)
                self._sfx(A.sfx_chime(), 0.5)
            self._sfx(A.sfx_coin(), 0.35)
        elif pts == 0:
            self._sfx(A.sfx_bad(), 0.25)


# =========================================================================== BULLES
class BubbleGame(_Game):
    DURATION = 40.0

    def __init__(self, text, audio, records):
        super().__init__(text, audio, records, "bulles")
        self.bubbles = []
        self.spawn_acc = 0.0
        self.t_start = None
        self.combo, self.last_pop = 0, -10.0
        self.best_combo = 0
        self.popped = 0
        self.w = self.h = 0
        self.shake = 0.0
        self._sprites = {}
        self.slow_until = self.fire_until = 0.0
        self.bonus_time = 0.0

    def _sprite(self, r, hue):
        key = (r, int(hue * 12))
        if key not in self._sprites:
            s = 2 * r + 4
            yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
            d = np.sqrt((xx - s / 2) ** 2 + (yy - s / 2) ** 2) / r
            rim = np.clip(1 - np.abs(d - 0.93) / 0.09, 0, 1)
            body = np.clip(1 - d, 0, 1) ** 0.5 * 0.25 * (d < 1)
            hl = np.clip(1 - np.sqrt((xx - s * 0.36) ** 2 + (yy - s * 0.32) ** 2) / (r * 0.28), 0, 1) ** 2
            alpha = np.clip(rim * 0.9 + body + hl * 0.9, 0, 1)[..., None]
            col = np.array(ui.hsv_color(key[1] / 12, 0.55, 1.0), np.float32)
            rgb = col * (1 - hl[..., None]) + 255 * hl[..., None]
            self._sprites[key] = (rgb.astype(np.float32), alpha.astype(np.float32))
        return self._sprites[key]

    def update(self, frame, tracker, dt):
        h, w = frame.shape[:2]
        self.w, self.h = w, h
        T = self.text
        now = time.monotonic()
        hands = tracker.hands()
        tips = []
        for hd in hands:
            tips.append((hd.pts[8], hd.palm * 0.35))
            tips.append((hd.pts[12], hd.palm * 0.3))
        e = now - self.t_phase

        if self.phase == "wait":
            ui.dim(frame, 0.6)
            T.draw(frame, "BULLES", (w // 2, ui.u(60)), 64, ui.WHITE, "mt", bold=True, stroke=3)
            T.draw(frame, "Éclate les bulles avec tes doigts · étoiles = 50 · évite les bombes · attrape les bonus",
                   (w // 2, ui.u(140)), 24, ui.WHITE, "mt", stroke=2)
            self.hold = self.hold + dt if hands else max(0.0, self.hold - dt)
            T.draw(frame, "Montre tes mains pour commencer" if not hands else "C'est parti…", (w // 2, h // 2),
                   30, ui.HUD_GREEN if hands else ui.WHITE, "mm", bold=True, stroke=2)
            ui.arc(frame, (w // 2, h // 2 + ui.u(80)), ui.u(30), self.hold / 0.8, ui.GREEN, 5)
            if self.hold >= 0.8:
                self._set("countdown")
        elif self.phase == "countdown":
            left = 3 - e
            if left <= 0:
                self._set("play")
                self.t_start = now
                self.start_music(0, 0.3)
            else:
                T.draw(frame, str(int(left) + 1), (w // 2, h // 2), 140, ui.WHITE, "mm", bold=True, stroke=4,
                       alpha=0.4 + 0.6 * (left - int(left)))
        elif self.phase == "play":
            self._play(frame, tips, dt, now)
        if self.phase == "end":
            self._end_screen(frame, [("Score", str(self.score)), ("Bulles éclatées", str(self.popped)),
                                     ("Meilleur combo", str(self.best_combo)),
                                     ("Record", str(self.records.get("bulles", 0)))])
        for p, r in tips:
            cv2.circle(frame, (int(p[0]), int(p[1])), max(ui.u(8), int(r * 0.5)), ui.WHITE, max(1, ui.u(2)),
                       cv2.LINE_AA)
        self.particles.update(dt)
        self.particles.draw(frame)
        self._draw_pops(frame)
        return None

    def _play(self, frame, tips, dt, now):
        h, w = frame.shape[:2]
        T = self.text
        el = now - self.t_start
        left = self.DURATION + self.bonus_time - el
        sprint = left < 10
        rate = (1.6 + el * 0.08) * (1.6 if sprint else 1.0)
        slow = 0.45 if now < self.slow_until else 1.0
        self.spawn_acc += rate * dt * slow
        while self.spawn_acc >= 1:
            self.spawn_acc -= 1
            kind = random.choices(["bulle", "étoile", "bombe", "gel", "temps", "feu"],
                                  [0.74, 0.09, 0.12 + el * 0.002, 0.02, 0.015, 0.02])[0]
            r = ui.u(random.uniform(38, 72) if kind == "bulle" else 40)
            self.bubbles.append({"kind": kind, "x": random.uniform(0.08, 0.92) * w, "y": h + r, "r": r,
                                 "vy": -ui.u(random.uniform(110, 210) + el * 3) * (1.4 if kind == "étoile" else 1),
                                 "ph": random.random() * 6, "hue": random.random(), "dead": False})
        for b in self.bubbles:
            b["y"] += b["vy"] * dt * slow
            b["x"] += math.sin(now * 2 + b["ph"]) * ui.u(40) * dt * slow
            for p, rr in tips:
                if math.hypot(p[0] - b["x"], p[1] - b["y"]) < b["r"] + rr:
                    self._pop(b, now)
                    break
        self.bubbles = [b for b in self.bubbles if not b["dead"] and b["y"] > -b["r"] * 2]
        if self.shake > 0:
            self.shake = max(0.0, self.shake - dt * 3)
            dx = int(math.sin(now * 60) * ui.u(18) * self.shake)
            frame[:] = np.roll(frame, dx, axis=1)
        for b in self.bubbles:
            if b["kind"] == "bulle":
                ui.blit(frame, self._sprite(b["r"], b["hue"]), (b["x"], b["y"]))
            else:
                icon = {"étoile": "⭐", "bombe": "💣", "gel": "❄️", "temps": "⏱️", "feu": "🔥"}[b["kind"]]
                if b["kind"] in ("gel", "temps", "feu"):
                    ui.blit(frame, self._sprite(b["r"] + ui.u(8), 0.55), (b["x"], b["y"]))
                ui.blit(frame, icons.get(icon, b["r"] * 2), (b["x"], b["y"]))
        if now < self.slow_until:
            frame[:] = cv2.addWeighted(frame, 0.85, np.full_like(frame, (255, 200, 120)), 0.15, 0)
        active = [(e_, u_) for e_, u_ in (("❄️", self.slow_until), ("🔥", self.fire_until)) if now < u_]
        for k, (ic, until) in enumerate(active):
            p = (w // 2 + (k - (len(active) - 1) / 2) * ui.u(90), ui.u(60))
            ui.blit(frame, icons.get(ic, ui.u(50)), p)
            ui.arc(frame, p, ui.u(34), (until - now) / 5.0, ui.WHITE, 4)
        if sprint:
            T.draw(frame, "SPRINT FINAL !", (w // 2, ui.u(110)), 30, (80, 200, 255), "mt", bold=True, stroke=3,
                   alpha=0.6 + 0.4 * math.sin(now * 10))
        # interface
        frac = max(0.0, left / self.DURATION)
        cv2.rectangle(frame, (0, 0), (int(w * frac), ui.u(8)), ui.HUD_GREEN if left > 8 else ui.RED, -1)
        T.draw(frame, str(self.score), (ui.u(24), ui.u(22)), 44, ui.WHITE, "lt", bold=True, stroke=3)
        T.draw(frame, f"{int(math.ceil(left))} s", (w - ui.u(24), ui.u(22)), 36, ui.WHITE, "rt", bold=True, stroke=3)
        if self.combo >= 3:
            T.draw(frame, f"combo x{self.combo}", (ui.u(24), ui.u(80)), 24, ui.HUD_GREEN, "lt", bold=True, stroke=2)
        if left <= 0:
            self._finish()

    def _pop(self, b, now):
        b["dead"] = True
        if b["kind"] == "bombe":
            self.score = max(0, self.score - 25)
            self.combo = 0
            self.shake = 1.0
            self.particles.emit((b["x"], b["y"]), 60, hue=0.02, speed=500)
            self._popup("-25", (b["x"], b["y"]), ui.RED)
            self._sfx(A.sfx_bad(), 0.5)
            return
        if b["kind"] in ("gel", "temps", "feu"):
            label = {"gel": "RALENTI !", "temps": "+5 s", "feu": "POINTS x2 !"}[b["kind"]]
            if b["kind"] == "gel":
                self.slow_until = now + 5
            elif b["kind"] == "temps":
                self.bonus_time += 5
            else:
                self.fire_until = now + 5
            self.particles.emit((b["x"], b["y"]), 80, hue=0.55 if b["kind"] == "gel" else 0.08, speed=450)
            self._popup(label, (b["x"], b["y"]), (255, 230, 120), 40)
            self._sfx(A.sfx_chime(), 0.5)
            return
        self.combo = self.combo + 1 if now - self.last_pop < 0.9 else 1
        self.best_combo = max(self.best_combo, self.combo)
        self.last_pop = now
        pts = (50 if b["kind"] == "étoile" else 10) * (1 + min(4, self.combo // 4)) * (2 if now < self.fire_until else 1)
        self.score += pts
        self.popped += 1
        self.particles.emit((b["x"], b["y"]), 30 if b["kind"] == "bulle" else 70, hue=b["hue"], speed=300)
        self._popup(f"+{pts}", (b["x"], b["y"]), (60, 220, 255) if b["kind"] == "étoile" else ui.WHITE)
        self._sfx(A.sfx_coin() if b["kind"] == "étoile" else A.sfx_pop(0.8 + min(self.combo, 10) * 0.08), 0.5)
