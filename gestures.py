"""Reconnaissance des gestes, version robuste.

Poses (par main, anti-rebond 3 détections sur 4) :
  open ✋  fist ✊  point ☝️  peace ✌️  rock 🤘  thumbs_up 👍  pinch 🤏

Événements :
  swipe   un glissé est analysé comme UN mouvement complet (début → fin), pas image par image :
          vitesse de démarrage, déplacement total, ligne droite, main pas fermée. Un seul événement par
          geste, et le mouvement de retour de la main est ignoré.
  hold    pose tenue (poing, pouce, rock, V, main ouverte immobile)
  heart   cœur avec les deux mains
  clap    applaudissement : pic sonore au micro + mains proches juste avant (ou, sans micro,
          mains qui se rejoignent très vite)
  pinch_tap  pince franche et brève (désactivée par défaut dans le mode live)
"""
import collections

import numpy as np

EMOJI = {"open": "✋", "fist": "✊", "point": "☝️", "peace": "✌️", "rock": "🤘", "thumbs_up": "👍",
         "pinch": "🤏", "vulcan": "🖖"}
HOLD_TIME = {"fist": 1.0, "thumbs_up": 0.6, "rock": 0.6, "peace": 0.8, "open": 1.2, "vulcan": 0.5}
STILL_POSES = ("open", "peace", "vulcan")  # poses qui doivent rester immobiles pendant le maintien

# glissés (unités : largeur d'image et secondes)
SWIPE_START = 0.5     # vitesse qui ouvre un mouvement
SWIPE_END = 0.25      # vitesse sous laquelle le mouvement est terminé
SWIPE_PEAK = 0.75     # pic de vitesse minimal
SWIPE_DIST = 0.15     # déplacement minimal
SWIPE_STRAIGHT = 1.8  # |dx| doit dépasser 1,8x |dy|


def _bend(P, a, b, c):
    """Angle de pliage (degrés) à l'articulation b : 0 = droit."""
    v1, v2 = P[b] - P[a], P[c] - P[b]
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


FINGERS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]


def finger_curls(hand):
    """Pliage total de chaque doigt (MCP + PIP + DIP), en 3D si dispo → indépendant de l'orientation."""
    P = hand.world if hand.world is not None else np.c_[hand.pts, np.zeros(21, np.float32)]
    curls = [_bend(P, 0, m, p) + _bend(P, m, p, d) + _bend(P, p, d, t) for m, p, d, t in FINGERS]
    thumb = _bend(P, 1, 2, 3) + _bend(P, 2, 3, 4)
    palm = float(np.linalg.norm(P[0] - P[9])) + 1e-9
    thumb_out = float(np.linalg.norm(P[4] - P[5])) / palm > 0.55 and float(np.linalg.norm(P[4] - P[9])) / palm > 0.6
    pinch3d = float(np.linalg.norm(P[4] - P[8])) / palm
    return curls, thumb, thumb_out, pinch3d, P


def classify(hand, pinch_thr):
    curls, thumb, thumb_out, pinch3d, P = finger_curls(hand)
    ext = [c < 75 for c in curls]
    fold = [c > 125 for c in curls]
    p = hand.pts
    if all(fold):
        # pouce levé : pouce tendu et pointé vers le haut de l'image
        up = p[2, 1] - p[4, 1]
        if thumb_out and thumb < 50 and up > 0.35 * hand.palm and up > abs(p[4, 0] - p[2, 0]):
            return "thumbs_up"
        return "fist"
    pinch = pinch3d < 0.33 if hand.world is not None else hand.pinch_ratio < pinch_thr
    if pinch and not fold[0]:
        return "pinch"
    if all(c < 100 for c in curls):  # doigts tendus (même un peu pliés) → salut vulcain ?
        # 🖖 salut vulcain : index+majeur collés, annulaire+auriculaire collés, grand écart au milieu
        dirs = [P[t] - P[m] for m, _, _, t in FINGERS]

        def gap(a, b):
            c_ = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            return float(np.degrees(np.arccos(np.clip(c_, -1, 1))))
        g_im, g_mr, g_rp = gap(dirs[0], dirs[1]), gap(dirs[1], dirs[2]), gap(dirs[2], dirs[3])
        # écarts mesurés DANS LA LARGEUR de la main (axe des jointures index → auriculaire) :
        # indépendant de la longueur des doigts (l'auriculaire est plus court) et de l'orientation
        ax = P[17] - P[5]
        mcpw = float(np.linalg.norm(ax)) + 1e-9
        ax = ax / mcpw
        lat = [float(np.dot(P[t] - P[0], ax)) for t in (8, 12, 16, 20)]
        l_im, l_mr, l_rp = lat[1] - lat[0], lat[2] - lat[1], lat[3] - lat[2]
        if (l_mr > 0.42 * mcpw and l_mr > 1.8 * max(l_im, l_rp, 0.05 * mcpw)) or \
                (g_mr > 12 and g_mr > 2.0 * g_im and g_mr > 1.5 * g_rp):
            return "vulcan"
        if all(ext):
            return "open"
    if ext[0] and fold[1] and fold[2] and fold[3]:
        return "point"
    if ext[0] and ext[1] and fold[2] and fold[3]:
        # V : index et majeur bien écartés (angle entre les deux doigts)
        v1, v2 = P[8] - P[5], P[12] - P[9]
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        if np.degrees(np.arccos(np.clip(cos, -1, 1))) > 10:
            return "peace"
    if ext[0] and ext[3] and fold[1] and fold[2]:
        return "rock"
    return "none"


class _HandState:
    def __init__(self, now):
        self.raw = collections.deque(maxlen=4)
        self.pose = "none"
        self.pose_t = now
        self.traj = collections.deque()
        self.first = now
        self.seen = now
        self.hold_fired = False
        self.pinch_t = None
        self.pinch_pos = None
        self.pinch_min = 9.0
        self.hand = None
        self.vel = np.zeros(2, np.float32)
        self.last_p = None
        self.last_t = None
        self.stroke = None


class GestureEngine:
    def __init__(self, pinch_thr=0.4, audio=None):
        self.pinch_thr = pinch_thr
        self.audio = audio
        self.st = {}
        self.last_swipe = (None, -10.0)
        self.close_t = -10.0
        self.two_t = -10.0
        self.clap_hist = collections.deque()
        self.last_clap = -10.0
        self.prev_clap = -10.0
        self._onset_seen = 0.0
        self._n_prev = 0
        self.heart_t = None
        self.last_heart = -10.0
        self.progress = []
        self.swipe_trail = []
        self._seq = -1

    def poses(self):
        return {k: (s.pose, s.hand) for k, s in self.st.items() if s.hand is not None}

    def update(self, hands, seq, now, size, allow=frozenset()):
        ev = []
        w, h = size
        if "clap" in allow:  # le pic sonore peut arriver entre deux détections
            self._clap_audio(now, ev)
        if seq == self._seq:
            return ev
        self._seq = seq
        keys = []
        labels = [hd.label for hd in hands]
        for i, hd in enumerate(hands):
            k = hd.label if labels.count(hd.label) == 1 else f"{hd.label}{i}"
            keys.append(k)
            s = self.st.get(k)
            if s is None:
                s = self.st[k] = _HandState(now)
            s.seen, s.hand = now, hd
            s.raw.append(classify(hd, self.pinch_thr))
            # "main ouverte au sens large" : au moins 3 doigts tendus (tolère le flou d'un geste rapide)
            s.openish = sum(c < 80 for c in finger_curls(hd)[0]) >= 3
            raw = list(s.raw)
            best = max(set(raw), key=raw.count)
            if raw.count(best) >= min(3, len(raw)) and best != s.pose:
                self._pose_change(s, best, now, ev, allow)
                s.pose, s.pose_t, s.hold_fired = best, now, False
            if s.pose == "pinch":
                s.pinch_min = min(s.pinch_min, hd.pinch_ratio)
            p = hd.pts[9].astype(np.float32)
            if s.last_p is not None and now > s.last_t:
                gap = now - s.last_t
                if gap > 0.15:  # main retrouvée après une perte : pas de vitesse fantôme
                    s.vel[:] = 0
                    s.stroke = None
                    s.traj.clear()
                else:
                    v = (p - s.last_p) / gap / w
                    if float(np.hypot(*v)) > 6.0:  # saut de détection (main "téléportée") : on oublie
                        s.vel[:] = 0
                        s.stroke = None
                        s.traj.clear()
                    else:
                        s.vel = 0.5 * s.vel + 0.5 * v
            s.last_p, s.last_t = p, now
            s.traj.append((now, float(p[0]), float(p[1])))
            while s.traj and now - s.traj[0][0] > 1.5:
                s.traj.popleft()
        for k in list(self.st):
            if k not in keys:
                s = self.st[k]
                if s.stroke is not None and "swipe" in allow:  # main perdue en plein glissé (flou)
                    self._end_stroke(s, now, w, h, ev, lost=True)
                s.hand = None
                if now - s.seen > 0.5:
                    del self.st[k]

        self.progress = []
        self.swipe_trail = []
        for k in keys:
            s = self.st[k]
            if "swipe" in allow:
                self._swipe(s, now, w, h, ev)
            else:
                s.stroke = None
            if "hold" in allow:
                self._hold(s, now, w, ev)
        if len(hands) < 2 and self._n_prev >= 2:
            self.close_t = now  # deux mains qui n'en font plus qu'une : elles se sont rejointes
        self._n_prev = len(hands)
        if len(hands) >= 2:
            self.two_t = now
            a, b = hands[0], hands[1]
            dist = float(np.linalg.norm(a.pts[9] - b.pts[9]))
            if dist < 0.35 * w:
                self.close_t = now
            if "heart" in allow:
                self._heart(hands, now, ev)
            if "clap" in allow and (self.audio is None or not self.audio.mic_ok):
                self._clap_visual(dist, now, w, (a.pts[9] + b.pts[9]) / 2, ev)
        else:
            self.heart_t = None
        return ev

    # ------------------------------------------------------------------ pince
    def _pose_change(self, s, new, now, ev, allow):
        hd = s.hand
        if new == "pinch":
            s.pinch_t, s.pinch_pos, s.pinch_min = now, hd.pinch_point.copy(), hd.pinch_ratio
        elif s.pose == "pinch" and s.pinch_t is not None:
            dur = now - s.pinch_t
            moved = float(np.linalg.norm(hd.pinch_point - s.pinch_pos))
            if ("pinch" in allow and 0.08 <= dur < 0.45 and moved < 0.6 * hd.palm
                    and s.pinch_min < 0.65 * self.pinch_thr and new in ("open", "none", "point")):
                ev.append(("pinch_tap", s, hd))
            s.pinch_t = None

    # ------------------------------------------------------------------ glissé
    def _swipe(self, s, now, w, h, ev):
        speed = float(np.hypot(*s.vel))
        if s.stroke is None:
            if speed > SWIPE_START and now - s.first > 0.15 and getattr(s, "openish", False) and len(s.traj) >= 3:
                # le geste a commencé un peu avant d'atteindre la vitesse de départ
                start = next(p for p in s.traj if now - p[0] <= 0.2)
                s.stroke = {"start": np.array(start[1:], np.float32), "t0": start[0], "peak": speed,
                            "n": 0, "good": 0}
            return
        st = s.stroke
        st["peak"] = max(st["peak"], speed)
        st["n"] += 1
        st["good"] += bool(getattr(s, "openish", False))
        cur = np.array(s.traj[-1][1:], np.float32)
        self.swipe_trail.append((st["start"], cur, min(1.0, abs(cur[0] - st["start"][0]) / (SWIPE_DIST * w))))
        if speed < SWIPE_END or now - st["t0"] > 0.7:
            self._end_stroke(s, now, w, h, ev)

    def _end_stroke(self, s, now, w, h, ev, lost=False):
        st, s.stroke = s.stroke, None
        if st is None or s.last_p is None:
            return
        dx, dy = (s.last_p - st["start"]).tolist()
        dur = now - st["t0"]
        if not (0.06 <= dur <= 0.8) or not (SWIPE_PEAK <= st["peak"] <= 7.0):  # > 7 l/s = saut de détection
            return
        if st["n"] < (1 if lost else 2) or st["good"] / max(1, st["n"]) < 0.5:
            return
        dist = SWIPE_DIST * (0.75 if lost else 1.0) * w
        if abs(dx) < dist or abs(dx) < SWIPE_STRAIGHT * abs(dy):
            return
        dirn = "left" if dx < 0 else "right"
        last_dir, last_t = self.last_swipe
        if dirn == last_dir and now - last_t < 0.4:
            return
        if last_dir and dirn != last_dir and now - last_t < 1.3 and (now - last_t < 0.35 or st["peak"] < 1.5):
            return  # retour de la main après un glissé (un vrai geste inverse doit être franc et rapide)
        self.last_swipe = (dirn, now)
        ev.append(("swipe", dirn))

    # ------------------------------------------------------------------ poses tenues
    def _hold(self, s, now, w, ev):
        """Poses tenues : un chronomètre par pose, qui tolère de courtes hésitations de la détection
        (< 0,3 s) sans repartir de zéro. Quand une pose se déclenche, les autres repartent à zéro."""
        timers = s.__dict__.setdefault("ht", {})  # pose -> [début, vue pour la dernière fois, déclenchée]
        cur = s.pose if s.pose in HOLD_TIME else None
        if cur is not None:
            tm = timers.get(cur)
            if tm is None or now - tm[1] > 0.3:
                timers[cur] = [now, now, False]
            else:
                tm[1] = now
            if cur in STILL_POSES:  # doit rester à peu près immobile
                tol = 0.035 if cur == "open" else 0.06
                recent = [p for p in s.traj if now - p[0] <= 0.5]
                if len(recent) > 1:
                    xs = [p[1] for p in recent]
                    ys = [p[2] for p in recent]
                    if max(xs) - min(xs) > tol * w or max(ys) - min(ys) > tol * w:
                        timers[cur][0] = now
        for p in [p for p, tm in timers.items() if now - tm[1] > 0.3]:
            del timers[p]
        best = None
        for p, (t0, _, fired) in timers.items():
            if fired:
                continue
            frac = (now - t0) / HOLD_TIME[p]
            if best is None or frac > best[1]:
                best = (p, frac)
        if best is None:
            return
        p, frac = best
        if now - timers[p][0] > 0.2:
            self.progress.append((s.hand.pts[9], min(1.0, frac), p))
        if frac >= 1.0:
            for tm in timers.values():
                tm[2] = True
            ev.append(("hold", p, s.hand))

    # ------------------------------------------------------------------ deux mains
    def _heart(self, hands, now, ev):
        a, b = hands[0], hands[1]
        pal = (a.palm + b.palm) / 2
        ok = (np.linalg.norm(a.pts[8] - b.pts[8]) < 0.55 * pal and np.linalg.norm(a.pts[4] - b.pts[4]) < 0.55 * pal
              and (a.pts[4, 1] + b.pts[4, 1]) / 2 - (a.pts[8, 1] + b.pts[8, 1]) / 2 > 0.5 * pal)
        if not ok or now - self.last_heart < 2.5:
            self.heart_t = None
            return
        if self.heart_t is None:
            self.heart_t = now
        el = now - self.heart_t
        c = (a.pts[8] + b.pts[8] + a.pts[4] + b.pts[4]) / 4
        self.progress.append((c, min(1.0, el / 0.5), "heart"))
        if el >= 0.5:
            self.last_heart, self.heart_t = now, None
            ev.append(("heart", c))

    def _clap_audio(self, now, ev):
        """Clap = claquement sec repéré au micro + deux mains vues et rapprochées juste avant.
        Deux claps espacés de 0,12 à 0,75 s = double clap."""
        if self.audio is None or not self.audio.mic_ok:
            return
        for o in [o for o in list(self.audio.onsets) if o > self._onset_seen]:
            self._onset_seen = o
            if not (o - self.two_t < 1.2 and o - self.close_t < 0.9):
                continue
            hs = [s.hand for s in self.st.values() if s.hand is not None]
            pos = np.mean([hd.pts[9] for hd in hs], axis=0) if hs else None
            self.last_clap = o
            ev.append(("clap", pos))
            if 0.12 <= o - self.prev_clap <= 0.75:
                ev.append(("double_clap", pos))
                self.prev_clap = -10.0
            else:
                self.prev_clap = o

    def _clap_visual(self, dist, now, w, pos, ev):
        self.clap_hist.append((now, dist))
        while self.clap_hist and now - self.clap_hist[0][0] > 0.5:
            self.clap_hist.popleft()
        if now - self.last_clap < 1.2:
            return
        if dist < 0.12 * w and max(d for _, d in self.clap_hist) > 0.32 * w:
            self.last_clap = now
            self.clap_hist.clear()
            ev.append(("clap", pos))
