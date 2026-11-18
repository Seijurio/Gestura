"""Caméra (thread dédié) + détection asynchrone MediaPipe (mains, visage, silhouette) + géométrie.

Tout tourne en parallèle : la caméra capture dans son thread, MediaPipe calcule dans les siens
(mode LIVE_STREAM, il saute les images quand il est occupé), et la boucle d'affichage prend
toujours le dernier résultat dispo → l'image reste fluide même si la détection rame.
"""
import itertools
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

SIG_IDX = [33, 133, 362, 263, 1, 168, 2, 61, 291, 152, 10, 234, 454, 70, 300]
_SIG_I, _SIG_J = np.triu_indices(len(SIG_IDX), 1)


# =========================================================================== données
@dataclass
class Hand:
    pts: np.ndarray  # (21, 2) pixels
    label: str
    world: np.ndarray = None  # (21, 3) mètres, repère de la main (pour les angles des doigts en 3D)

    @property
    def palm(self) -> float:
        return float(np.linalg.norm(self.pts[WRIST] - self.pts[MIDDLE_MCP])) + 1e-6

    @property
    def pinch_ratio(self) -> float:
        return float(np.linalg.norm(self.pts[THUMB_TIP] - self.pts[INDEX_TIP])) / self.palm

    @property
    def pinch_point(self):
        return (self.pts[THUMB_TIP] + self.pts[INDEX_TIP]) / 2

    def _ext(self, tip, pip, k=1.12):
        w = self.pts[WRIST]
        return np.linalg.norm(self.pts[tip] - w) > k * np.linalg.norm(self.pts[pip] - w)

    def _fold(self, tip, pip):
        w = self.pts[WRIST]
        return np.linalg.norm(self.pts[tip] - w) < np.linalg.norm(self.pts[pip] - w)

    @property
    def is_open(self) -> bool:
        return all(self._ext(t, p, 1.1) for t, p in ((8, 6), (12, 10), (16, 14), (20, 18)))

    @property
    def is_peace(self) -> bool:
        """✌️ index + majeur tendus et écartés, annulaire + auriculaire pliés."""
        return (self._ext(8, 6) and self._ext(12, 10) and self._fold(16, 14) and self._fold(20, 18)
                and np.linalg.norm(self.pts[8] - self.pts[12]) > 0.3 * self.palm)


@dataclass
class Face:
    pts: np.ndarray  # (478, 3) pixels

    @property
    def center(self):
        p = self.pts
        return np.array([(p[234, 0] + p[454, 0]) / 2, (p[10, 1] + p[152, 1]) / 2], np.float32)

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.pts[454, :2] - self.pts[234, :2])) + 1e-6

    @property
    def height(self) -> float:
        return float(np.linalg.norm(self.pts[152, :2] - self.pts[10, :2])) + 1e-6

    @property
    def yaw(self) -> float:
        return float((self.pts[1, 0] - self.center[0]) / (self.width / 2))

    @property
    def pitch(self) -> float:
        return float((self.pts[1, 1] - self.center[1]) / (self.height / 2))

    @property
    def signature(self):
        p = self.pts
        iod = np.linalg.norm(p[33] - p[263]) + 1e-6
        s = p[SIG_IDX]
        return np.linalg.norm(s[_SIG_I] - s[_SIG_J], axis=1) / iod

    def box(self, sx=1.35, sy=1.55):
        """Quadrilatère (TL, TR, BR, BL) qui suit le visage, incliné comme la tête."""
        p = self.pts[:, :2]
        ux = p[263] - p[33]
        if ux[0] < 0:
            ux = -ux
        ux = ux / (np.linalg.norm(ux) + 1e-6)
        uy = np.array([-ux[1], ux[0]], np.float32)
        c = self.center - uy * 0.06 * self.height
        hw, hh = self.width * sx / 2, self.height * sy / 2
        return np.array([c - ux * hw - uy * hh, c + ux * hw - uy * hh,
                         c + ux * hw + uy * hh, c - ux * hw + uy * hh], np.float32)


@dataclass
class Body:
    pts: np.ndarray  # (33, 2) pixels — 11/12 épaules, 13/14 coudes, 15/16 poignets, 23/24 hanches
    vis: np.ndarray  # (33,) visibilité 0..1 (baisse quand le point est "tenu" faute de mesure fiable)


class BodyFilter:
    """Filtre anti-sauts pour le squelette :
    - un point peu visible garde sa dernière bonne position (jusqu'à 0,5 s) au lieu de partir n'importe où ;
    - un grand saut soudain n'est accepté que s'il est confirmé par l'image suivante ;
    - puis lissage One Euro (stable à l'arrêt, réactif en mouvement)."""

    def __init__(self):
        self.x = None
        self.last_ok = np.zeros(33)
        self.pending = np.full((33, 2), np.nan, np.float32)
        self.euro = OneEuro(1.3, 0.004)

    def reset(self):
        self.x = None
        self.euro.reset()

    def __call__(self, xy, vis, t):
        if self.x is None:
            self.x = xy.copy()
            self.last_ok[:] = t
            return self.euro(xy, t), vis
        sw = max(40.0, float(np.linalg.norm(self.x[11] - self.x[12])))
        d = np.linalg.norm(xy - self.x, axis=1)
        good = vis >= 0.5
        jump = d > 0.6 * sw
        confirm = np.linalg.norm(xy - self.pending, axis=1) < 0.25 * sw
        accept = good & (~jump | confirm)
        self.pending = np.where((good & jump & ~confirm)[:, None], xy, np.nan).astype(np.float32)
        stale = t - self.last_ok > 0.5
        accept |= stale & (vis >= 0.3)   # ne pas rester figé indéfiniment
        self.last_ok[accept] = t
        target = np.where(accept[:, None], xy, self.x)
        self.x = self.euro(target, t).copy()
        q = np.clip(1.0 - (t - self.last_ok) / 0.5, 0, 1)
        return self.x, np.minimum(np.maximum(vis, 0.6 * q * (vis >= 0.2)), 1.0) * np.where(accept, 1.0, 0.85)


def signature_distance(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.abs(a - b)) / (np.mean(b) + 1e-6))


# =========================================================================== lissage / géométrie
class Smoother:
    """Lissage adaptatif : fort quand ça bouge peu (anti-tremblement), quasi nul quand ça bouge vite."""

    def __init__(self, a_min=0.25, a_max=0.95, speed_ref=35.0):
        self.a_min, self.a_max, self.speed_ref = a_min, a_max, speed_ref
        self.value = None

    def reset(self):
        self.value = None

    def update(self, pts):
        pts = np.asarray(pts, np.float32)
        if self.value is None or self.value.shape != pts.shape:
            self.value = pts.copy()
            return self.value
        d = np.linalg.norm(pts - self.value, axis=-1, keepdims=True)
        a = np.clip(self.a_min + (self.a_max - self.a_min) * d / self.speed_ref, self.a_min, self.a_max)
        self.value = self.value + a * (pts - self.value)
        return self.value


class OneEuro:
    """Filtre « One Euro » : quasi aucun tremblement à l'arrêt, quasi aucun retard en mouvement."""

    def __init__(self, min_cutoff=1.4, beta=0.012, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    def reset(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, np.float32)
        if self.x is None or self.x.shape != x.shape:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x
        dt = max(1e-3, t - self.t)
        self.t = t
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        speed = np.linalg.norm(self.dx, axis=-1, keepdims=True)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x = self.x + a * (x - self.x)
        return self.x


def match_order(prev, pts):
    if prev is None or len(prev) != len(pts):
        return pts
    best, best_d = pts, None
    for perm in itertools.permutations(range(len(pts))):
        cand = pts[list(perm)]
        d = float(np.sum(np.linalg.norm(cand - prev, axis=1)))
        if best_d is None or d < best_d:
            best, best_d = cand, d
    return best


def convex_order(pts):
    c = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]


def order_tlbr(pts):
    """Ordonne 4 points en haut-gauche, haut-droite, bas-droite, bas-gauche (sens horaire écran)."""
    q = convex_order(np.asarray(pts, np.float32))
    return np.roll(q, -int(np.argmin(q[:, 0] + q[:, 1])), axis=0)


def quad_area(q):
    q = np.asarray(q, np.float32)
    return abs(float(np.cross(q[2] - q[0], q[3] - q[1]))) / 2


# =========================================================================== caméra
class Camera:
    def __init__(self, size=(1920, 1080), fps=60):
        """Ouvre la caméra et prend la meilleure résolution qu'elle fournit vraiment
        (on demande 4K, puis 1440p, puis 1080p : certaines caméras retombent sur une petite
        résolution par défaut quand on leur demande un format qu'elles n'ont pas)."""
        self.cap = None
        cands = [size, (1280, 720)]  # formats 16:9 natifs ; on garde la meilleure vraie résolution
        f = None
        for idx in (0, 1, 2):
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                continue
            best, best_frame, first = None, None, True
            for cw, ch in cands:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
                cap.set(cv2.CAP_PROP_FPS, fps)
                got = None
                t0 = time.monotonic()
                while time.monotonic() - t0 < (5 if first else 1.5):  # 1re fois : le temps d'autoriser
                    ok, fr = cap.read()
                    if ok and fr is not None:
                        got = fr
                        if fr.shape[1] >= 0.95 * cw:
                            break
                    time.sleep(0.03)
                first = False
                if got is not None and (best is None or got.shape[1] > best[0]):
                    best, best_frame = (got.shape[1], got.shape[0]), got
                if got is not None and got.shape[1] >= 0.95 * cw:
                    break
            if best is not None:
                if (best_frame.shape[1], best_frame.shape[0]) != (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, best[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best[1])
                self.cap, f = cap, best_frame
                break
            cap.release()
        if self.cap is None:
            raise RuntimeError("camera")
        self.frame, self.seq = f, 1
        self.size = (f.shape[1], f.shape[0])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        last = None
        while self._running:
            ok, f = self.cap.read()
            if not ok or f is None:
                time.sleep(0.005)
                continue
            now = time.monotonic()
            if last is not None and now > last:  # cadence réelle de la caméra (mesurée, pas annoncée)
                self.real_fps = 0.9 * getattr(self, "real_fps", 1 / (now - last)) + 0.1 / (now - last)
            last = now
            with self._cond:
                self.frame, self.seq = f, self.seq + 1
                self._cond.notify_all()

    def latest(self):
        """Dernière image disponible, sans attendre : (seq, frame)."""
        with self._cond:
            return self.seq, self.frame

    def read(self, last_seq, timeout=1.0):
        """Attend une image plus récente que last_seq. Renvoie (seq, frame) ou (last_seq, None)."""
        with self._cond:
            if self.seq <= last_seq:
                self._cond.wait(timeout)
            if self.seq <= last_seq:
                return last_seq, None
            return self.seq, self.frame

    def close(self):
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()


# =========================================================================== détection
class AsyncTracker:
    def __init__(self, models: Path, detect_width=960, on_progress=None):
        say = on_progress or (lambda *_: None)
        self.detect_width = detect_width
        self.size = (1280, 720)
        self._lock = threading.Lock()
        self._hands, self._hands_t = [], 0.0
        self.hands_seq = 0
        self._face, self._face_t = None, 0.0
        self._mask, self._mask_t = None, 0.0
        self._hsm = [OneEuro(), OneEuro()]
        self._body, self._body_t = None, 0.0
        self._hn = 0
        self.det_fps = 0.0
        self._last_hand_cb = None
        self._t0 = time.monotonic()
        self._last_ts = -1
        LIVE = vision.RunningMode.LIVE_STREAM

        say("hands")
        self.hand_lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models / "hand_landmarker.task")),
            running_mode=LIVE, num_hands=2,
            min_hand_detection_confidence=0.55, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5, result_callback=self._on_hands))
        say("face")
        self.face_lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models / "face_landmarker.task")),
            running_mode=LIVE, num_faces=1, result_callback=self._on_face))
        say("pose")
        pose_model = models / "pose_landmarker_heavy.task"  # 3-4x plus stable que "full", ~12 ms sur Apple Silicon
        if not pose_model.exists():
            pose_model = models / "pose_landmarker_full.task"
        self.pose_lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model)),
            running_mode=LIVE, num_poses=1, min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6, min_tracking_confidence=0.6, result_callback=self._on_pose))
        self._bsm = BodyFilter()
        say("seg")
        self.segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(models / "selfie_segmenter.tflite")),
            running_mode=LIVE, output_confidence_masks=True, output_category_mask=False,
            result_callback=self._on_seg))

    # ------------------------------------------------------------------ envoi
    def _ts(self):
        ts = max(int((time.monotonic() - self._t0) * 1000), self._last_ts + 1)
        self._last_ts = ts
        return ts

    def submit(self, frame, hands=True, face=False, seg=False, pose=False):
        if not (hands or face or seg or pose):
            return
        h, w = frame.shape[:2]
        self.size = (w, h)
        dw = min(self.detect_width, w)
        small = cv2.resize(frame, (dw, int(h * dw / w)), interpolation=cv2.INTER_AREA)
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        ts = self._ts()
        if hands:
            self.hand_lm.detect_async(img, ts)
        if face:
            self.face_lm.detect_async(img, ts)
        if seg:
            self.segmenter.segment_async(img, ts)
        if pose:
            self.pose_lm.detect_async(img, ts)

    # ------------------------------------------------------------------ callbacks (threads MediaPipe)
    def _on_hands(self, res, _img, _ts):
        w, h = self.size
        hands = []
        worlds = res.hand_world_landmarks or [None] * len(res.hand_landmarks)
        for lms, hd, wl in zip(res.hand_landmarks, res.handedness, worlds):
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms], np.float32)
            world = np.array([[p.x, p.y, p.z] for p in wl], np.float32) if wl else None
            hands.append(Hand(pts, hd[0].category_name if hd else "?", world))
        hands.sort(key=lambda x: x.pts[WRIST, 0])
        now = time.monotonic()
        with self._lock:
            if len(hands) != self._hn:
                for s in self._hsm:
                    s.reset()
                self._hn = len(hands)
            for i, hd in enumerate(hands[:2]):
                hd.pts = self._hsm[i](hd.pts, _ts / 1000.0).copy()
            self._hands, self._hands_t = hands, now
            self.hands_seq += 1
            if self._last_hand_cb is not None:
                dt = now - self._last_hand_cb
                if dt > 0:
                    self.det_fps = 0.9 * self.det_fps + 0.1 / dt if self.det_fps else 1 / dt
            self._last_hand_cb = now

    def _on_face(self, res, _img, _ts):
        w, h = self.size
        face = None
        if res.face_landmarks:
            face = Face(np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in res.face_landmarks[0]],
                                 np.float32))
        with self._lock:
            self._face, self._face_t = face, time.monotonic()

    def _on_pose(self, res, _img, ts):
        w, h = self.size
        body = None
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            raw = np.array([[lm.x * w, lm.y * h] for lm in lms], np.float32)
            vis = np.array([min(lm.visibility if lm.visibility is not None else 1.0,
                                lm.presence if lm.presence is not None else 1.0) for lm in lms], np.float32)
            # un point hors de l'image n'est pas fiable
            inside = (raw[:, 0] > -0.02 * w) & (raw[:, 0] < 1.02 * w) & (raw[:, 1] > -0.02 * h) & (raw[:, 1] < 1.02 * h)
            vis = np.where(inside, vis, vis * 0.3)
            xy, vis = self._bsm(raw, vis, ts / 1000.0)
            body = Body(xy.copy(), vis.astype(np.float32))
        else:
            self._bsm.reset()
        with self._lock:
            self._body, self._body_t = body, time.monotonic()

    def _on_seg(self, res, _img, _ts):
        if not res.confidence_masks:
            return
        m = res.confidence_masks[-1].numpy_view()
        m = np.array(m[..., 0] if m.ndim == 3 else m, np.float32)
        with self._lock:
            self._mask, self._mask_t = m, time.monotonic()

    # ------------------------------------------------------------------ lecture
    def hands(self, max_age=0.25):
        with self._lock:
            fresh = time.monotonic() - self._hands_t < max_age
            return [Hand(h.pts.copy(), h.label, h.world) for h in self._hands] if fresh else []

    def face(self, max_age=0.35):
        with self._lock:
            return self._face if time.monotonic() - self._face_t < max_age else None

    def body(self, max_age=0.35):
        with self._lock:
            return self._body if time.monotonic() - self._body_t < max_age else None

    def mask(self, max_age=0.6):
        with self._lock:
            return self._mask if time.monotonic() - self._mask_t < max_age else None

    def warmup(self, frame, timeout=4.0):
        """Premières inférences (les plus lentes) avant d'afficher quoi que ce soit."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self.submit(frame, hands=True, face=True, seg=True, pose=True)
            time.sleep(0.03)
            with self._lock:
                if self._hands_t and self._face_t and self._mask_t:
                    break

    def close(self):
        for t in (self.hand_lm, self.face_lm, self.segmenter, self.pose_lm):
            try:
                t.close()
            except Exception:
                pass
