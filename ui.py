"""Dessin : texte (Pillow, accents OK), anneaux, coche animée, cadre, réticule, squelette, particules.

Toutes les tailles sont pensées pour du 720p puis multipliées par S (= hauteur / 720),
donc l'interface reste nette et proportionnée en 1080p / 4K.
"""
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (100, 225, 90)
HUD_GREEN = (90, 255, 140)
GREY = (110, 110, 110)
RED = (80, 80, 245)

S = 1.0
CAM_FPS = 0.0  # cadence réelle de la caméra (affichée dans le HUD)
RING_LEVEL = 0.0  # puissance de la lumière d'appoint (pour le badge du HUD)


def set_scale(s):
    global S
    S = s


def u(v):
    return int(round(v * S))


def lerp(a, b, t):
    return a + (b - a) * t


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    x = min(max(x, 0.0), 1.0)
    return 3 * x * x - 2 * x * x * x


def approach(cur, target, dt, speed=10.0):
    """Interpolation exponentielle indépendante du framerate."""
    return cur + (target - cur) * (1 - math.exp(-speed * dt))


# =========================================================================== texte
_FONT_FILES = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def _load_font(size, bold):
    for path in _FONT_FILES:
        try:
            for idx in range(16):
                f = ImageFont.truetype(path, size, index=idx)
                style = f.getname()[1].lower()
                if (bold and style == "bold") or (not bold and style in ("regular", "medium", "roman")):
                    return f
        except OSError:
            continue
    try:
        return ImageFont.truetype(_FONT_FILES[-1], size)
    except OSError:
        return ImageFont.load_default(size=size)


class Text:
    def __init__(self):
        self._fonts, self._cache = {}, {}

    def _font(self, size, bold):
        k = (size, bold)
        if k not in self._fonts:
            self._fonts[k] = _load_font(size, bold)
        return self._fonts[k]

    def _render(self, text, size, color, bold, stroke):
        key = (text, size, color, bold, stroke)
        if key not in self._cache:
            if len(self._cache) > 600:
                self._cache.clear()
            f = self._font(size, bold)
            l, t, r, b = f.getbbox(text, stroke_width=stroke)
            im = Image.new("RGBA", (max(1, r - l + 2), max(1, b - t + 2)), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((1 - l, 1 - t), text, font=f, fill=(color[2], color[1], color[0], 255),
                                    stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            a = np.asarray(im).astype(np.float32) / 255.0
            self._cache[key] = (a[..., 2::-1] * 255.0, a[..., 3:4])
        return self._cache[key]

    def size(self, text, size=22, bold=False, stroke=0):
        bgr, a = self._render(text, max(8, u(size)), WHITE, bold, u(stroke))
        return a.shape[1], a.shape[0]

    def draw(self, img, text, pos, size=22, color=WHITE, anchor="lt", bold=False, stroke=0, alpha=1.0):
        if not text or alpha <= 0.01:
            return 0, 0
        bgr, a = self._render(text, max(8, u(size)), tuple(int(c) for c in color), bold, u(stroke))
        h, w = a.shape[:2]
        x, y = int(pos[0]), int(pos[1])
        x -= {"l": 0, "m": w // 2, "r": w}[anchor[0]]
        y -= {"t": 0, "m": h // 2, "b": h}[anchor[1]]
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x0 < x1 and y0 < y1:
            sa = a[y0 - y:y1 - y, x0 - x:x1 - x] * alpha
            roi = img[y0:y1, x0:x1].astype(np.float32)
            img[y0:y1, x0:x1] = (roi + (bgr[y0 - y:y1 - y, x0 - x:x1 - x] - roi) * sa).astype(np.uint8)
        return w, h


_TXT = None


def Text_default():
    """Instance de Text partagée (pour les modules qui n'en reçoivent pas)."""
    global _TXT
    if _TXT is None:
        _TXT = Text()
    return _TXT


def panel(img, x0, y0, x1, y1, alpha=0.6, radius=14):
    """Rectangle sombre arrondi semi-transparent."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    r = min(u(radius), (x1 - x0) // 2, (y1 - y0) // 2)
    cv2.rectangle(mask, (r, 0), (x1 - x0 - r, y1 - y0), 255, -1)
    cv2.rectangle(mask, (0, r), (x1 - x0, y1 - y0 - r), 255, -1)
    for cx, cy in ((r, r), (x1 - x0 - r, r), (r, y1 - y0 - r), (x1 - x0 - r, y1 - y0 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    a = (mask.astype(np.float32) / 255 * alpha)[..., None]
    roi[:] = (roi * (1 - a)).astype(np.uint8)


# =========================================================================== formes
def arc(img, center, r, progress, color, thickness=6, start=-90):
    if progress > 0.002:
        cv2.ellipse(img, (int(center[0]), int(center[1])), (int(r), int(r)), 0, start,
                    start + 360 * min(progress, 1.0), color, max(1, u(thickness)), cv2.LINE_AA)


def checkmark(img, center, size, progress, color=GREEN, thickness=8):
    cx, cy = center
    pts = np.array([[-0.45, 0.02], [-0.12, 0.33], [0.48, -0.32]]) * size + [cx, cy]
    s1, s2 = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1])
    L = ease_out(progress) * (s1 + s2)
    th = max(1, u(thickness))
    if L <= 0:
        return
    if L <= s1:
        p = pts[0] + (pts[1] - pts[0]) * (L / s1)
        cv2.line(img, tuple(pts[0].astype(int)), tuple(p.astype(int)), color, th, cv2.LINE_AA)
    else:
        cv2.line(img, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), color, th, cv2.LINE_AA)
        p = pts[1] + (pts[2] - pts[1]) * ((L - s1) / s2)
        cv2.line(img, tuple(pts[1].astype(int)), tuple(p.astype(int)), color, th, cv2.LINE_AA)


def ticks_ring(img, center, r_in, r_out, levels, color_on=GREEN, color_off=GREY, highlight=None):
    """Anneau de graduations façon Face ID. levels : 0..1 par graduation (bool accepté)."""
    levels = np.asarray(levels, np.float32)
    n = len(levels)
    cx, cy = center
    th = max(2, u(4))
    for i in range(n):
        a = 2 * math.pi * i / n
        ca, sa = math.cos(a), math.sin(a)
        lv = float(levels[i])
        boost = 0.0
        if highlight is not None:
            da = abs((a - highlight + math.pi) % (2 * math.pi) - math.pi)
            boost = max(0.0, 1 - da / 0.4) * u(16)
        ro = r_out + lv * u(8) + boost
        col = tuple(int(lerp(color_off[k], color_on[k], lv)) for k in range(3))
        cv2.line(img, (int(cx + ca * r_in), int(cy + sa * r_in)), (int(cx + ca * ro), int(cy + sa * ro)),
                 col, th, cv2.LINE_AA)


def soft_blur(img, sigma):
    """Flou gaussien de qualité et rapide : on floute une version réduite juste ce qu'il faut
    (le signal reste lisse, donc pas d'effet « blocs » à l'agrandissement)."""
    h, w = img.shape[:2]
    f = 1
    while sigma / (f * 2) >= 2.5 and min(h, w) // (f * 2) > 16:
        f *= 2
    if f == 1:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def feather_mask(h, w, draw_fn, feather):
    """Masque flottant 0..1 aux bords adoucis. draw_fn(mask_u8, scale) dessine la forme en blanc."""
    f = 4
    m = np.zeros((h // f, w // f), np.uint8)
    draw_fn(m, 1.0 / f)
    m = cv2.GaussianBlur(m, (0, 0), max(0.6, feather / f))
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) * (1 / 255.0)


def focus_circle(frame, center, r, dim=0.3):
    """Fond flouté + assombri, disque net au centre avec un bord doux (écran visage)."""
    h, w = frame.shape[:2]
    bg = soft_blur(frame, 18 * S)
    cv2.convertScaleAbs(bg, bg, alpha=dim)
    m = feather_mask(h, w, lambda mk, k: cv2.circle(mk, (int(center[0] * k), int(center[1] * k)),
                                                     int(r * k), 255, -1, cv2.LINE_AA), u(3))
    frame[:] = cv2.blendLinear(frame, bg, m, 1.0 - m)


def dim(frame, factor):
    cv2.convertScaleAbs(frame, frame, alpha=factor, beta=0)


def draw_hand(img, hand, color=WHITE, alpha=0.9, glow=True):
    from tracking import HAND_CONNECTIONS
    over = img.copy()
    p = hand.pts.astype(int)
    th = max(1, u(2))
    for a, b in HAND_CONNECTIONS:
        cv2.line(over, tuple(p[a]), tuple(p[b]), color, th, cv2.LINE_AA)
    for i, q in enumerate(p):
        tip = i in (4, 8)
        cv2.circle(over, tuple(q), u(7) if tip else u(3), GREEN if tip else color, -1, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)


def draw_quad(img, quad, color=WHITE):
    q = np.asarray(quad).astype(np.int32)
    cv2.polylines(img, [q], True, color, max(1, u(2)), cv2.LINE_AA)
    for p in q:
        cv2.circle(img, tuple(p), u(6), color, -1, cv2.LINE_AA)


def crosshair(img, c, size=20):
    x, y = int(c[0]), int(c[1])
    s = u(size)
    for col, th in ((BLACK, max(2, u(4))), (WHITE, max(1, u(1.4)))):
        cv2.line(img, (x - s, y), (x + s, y), col, th, cv2.LINE_AA)
        cv2.line(img, (x, y - s), (x, y + s), col, th, cv2.LINE_AA)


def step_dots(img, center, current, total):
    cx, cy = center
    gap = u(22)
    x0 = cx - (total - 1) * gap / 2
    for i in range(total):
        col = GREEN if i < current else (WHITE if i == current else GREY)
        cv2.circle(img, (int(x0 + i * gap), int(cy)), u(6) if i == current else u(5), col, -1, cv2.LINE_AA)


def spinner(img, center, r, t, color=WHITE):
    for k in range(12):
        a = 2 * math.pi * k / 12 + t * 6
        al = ((k / 12) + t * 0.95) % 1.0
        c = tuple(int(v * (0.25 + 0.75 * al)) for v in color)
        p1 = (int(center[0] + math.cos(a) * r * 0.55), int(center[1] + math.sin(a) * r * 0.55))
        p2 = (int(center[0] + math.cos(a) * r), int(center[1] + math.sin(a) * r))
        cv2.line(img, p1, p2, c, max(2, u(4)), cv2.LINE_AA)


def hsv_color(hue, s=1.0, v=1.0):
    """hue 0..1 → BGR"""
    h = (hue % 1.0) * 6
    i, f = int(h), h - int(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return (int(b * 255), int(g * 255), int(r * 255))


# =========================================================================== particules
class Particles:
    def __init__(self, maxn=600):
        self.p = np.zeros((0, 7), np.float32)  # x, y, vx, vy, life, max_life, hue
        self.maxn = maxn

    def emit(self, pos, n, hue=None, speed=140.0, spread=math.pi * 2, direction=-math.pi / 2):
        if n <= 0:
            return
        a = direction + (np.random.rand(n) - 0.5) * spread
        sp = speed * S * (0.3 + np.random.rand(n))
        life = 0.5 + np.random.rand(n) * 0.7
        hues = (np.full(n, hue) if hue is not None else np.random.rand(n)) + np.random.rand(n) * 0.08
        new = np.stack([np.full(n, pos[0]), np.full(n, pos[1]), np.cos(a) * sp, np.sin(a) * sp,
                        life, life, hues], axis=1).astype(np.float32)
        self.p = np.concatenate([self.p, new])[-self.maxn:]

    def update(self, dt):
        if not len(self.p):
            return
        p = self.p
        p[:, 0] += p[:, 2] * dt
        p[:, 1] += p[:, 3] * dt
        p[:, 3] += 260 * S * dt   # gravité
        p[:, 2] *= 0.985
        p[:, 4] -= dt
        self.p = p[p[:, 4] > 0]

    def draw(self, img):
        if not len(self.p):
            return
        h, w = img.shape[:2]
        layer = np.zeros((h // 2, w // 2, 3), np.uint8)
        rad = max(1.0, u(4) / 2)
        for x, y, _, _, life, ml, hue in self.p:
            k = life / ml
            cv2.circle(layer, (int(x / 2), int(y / 2)), max(1, int(rad * k + 0.5)),
                       hsv_color(hue, 0.6 * k, 1.0), -1, cv2.LINE_AA)
        glow = soft_blur(layer, u(5))
        layer = cv2.addWeighted(layer, 1.0, glow, 1.5, 0)
        cv2.add(img, cv2.resize(layer, (w, h), interpolation=cv2.INTER_LINEAR), img)


# =========================================================================== sprites / emojis
_EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
_sprites = {}


def emoji_rgba(char):
    """Emoji Apple (iOS) en couleur, à sa résolution maximale (160 px), recadré, RGBA uint8."""
    base = _sprites.get((char, "base"))
    if base is None:
        try:
            f = ImageFont.truetype(_EMOJI_FONT, 160)
            im = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
            ImageDraw.Draw(im).text((110, 110), char, font=f, embedded_color=True, anchor="mm")
            base = np.asarray(im.crop(im.getbbox() or (0, 0, 220, 220)))
        except OSError:
            base = np.zeros((8, 8, 4), np.uint8)
        _sprites[(char, "base")] = base
    return base


def resize_rgba(rgba, size):
    """Redimensionnement en alpha prémultiplié (pas de liseré sombre sur les bords) ;
    Lanczos pour agrandir (net), INTER_AREA pour réduire (sans crénelage)."""
    w, h = size
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    pm = np.concatenate([rgba[..., :3].astype(np.float32) * a, a * 255.0], axis=2)
    up = w > rgba.shape[1]
    out = cv2.resize(pm, (w, h), interpolation=cv2.INTER_LANCZOS4 if up else cv2.INTER_AREA)
    out = np.clip(out, 0, 255)
    al = out[..., 3:4] / 255.0
    rgb = np.where(al > 1e-3, out[..., :3] / np.maximum(al, 1e-3), 0)
    return np.clip(rgb, 0, 255), al


def emoji(char, size):
    """Emoji Apple → (bgr float32, alpha float32) mis en cache par taille."""
    size = max(8, int(size))
    key = (char, size)
    if key not in _sprites:
        base = emoji_rgba(char)
        hh, ww = base.shape[:2]
        sc = size / max(hh, ww)
        rgb, al = resize_rgba(base, (max(1, int(ww * sc)), max(1, int(hh * sc))))
        _sprites[key] = (rgb[..., 2::-1].astype(np.float32), al.astype(np.float32))
        if len(_sprites) > 400:
            _sprites.clear()
    return _sprites[key]


def blit(img, sprite, center, opacity=1.0):
    bgr, a = sprite
    h, w = a.shape[:2]
    x, y = int(center[0] - w / 2), int(center[1] - h / 2)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1 or opacity <= 0.01:
        return
    sa = a[y0 - y:y1 - y, x0 - x:x1 - x] * opacity
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi + (bgr[y0 - y:y1 - y, x0 - x:x1 - x] - roi) * sa).astype(np.uint8)


def random_hue():
    return random.random()
