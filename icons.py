"""Icônes maison, dessinées par le code (aucune image externe, aucun emoji).

Chaque icône est dessinée une fois sur une toile 1024x1024 (sur-échantillonnage → bords très lisses),
avec dégradés, reflets, contours et ombre, puis réduite à la taille demandée et mise en cache.
sprite(nom, taille) renvoie (bgr float32, alpha float32), le même format que ui.emoji → ui.blit.
"""
import math

import cv2
import numpy as np

N = 1024
_base = {}
_sized = {}
APP_ICONS = ("icon_danse", "icon_bulles", "icon_tranche", "icon_jacques", "icon_esquive")


# =========================================================================== toile
class Canvas:
    def __init__(self):
        self.rgb = np.zeros((N, N, 3), np.float32)
        self.a = np.zeros((N, N), np.float32)
        yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
        self.xx, self.yy = xx / N, yy / N

    @staticmethod
    def mask():
        return np.zeros((N, N), np.uint8)

    def put(self, mask, color, alpha=1.0):
        m = mask.astype(np.float32) / 255.0 * alpha
        col = np.asarray(color, np.float32)
        if col.ndim == 1:
            col = col[None, None, :]
        self.rgb = self.rgb * (1 - m[..., None]) + col * m[..., None]
        self.a = self.a + m * (1 - self.a)

    def shade(self, base, cx, cy, r, light=(-0.35, -0.4), strength=0.55, rim=0.25):
        """Dégradé « volume » : clair vers la lumière, sombre sur le bord opposé."""
        lx, ly = cx + light[0] * r, cy + light[1] * r
        d = np.sqrt((self.xx - lx) ** 2 + (self.yy - ly) ** 2) / (r * 1.6)
        k = np.clip(1.18 - strength * d, 0.35, 1.25)
        e = np.sqrt((self.xx - cx) ** 2 + (self.yy - cy) ** 2) / r
        k *= 1 - rim * np.clip((e - 0.75) / 0.25, 0, 1)
        return np.clip(np.asarray(base, np.float32)[None, None, :] * k[..., None], 0, 255)

    def shadow(self, mask, dx=0.012, dy=0.02, blur=18, alpha=0.35):
        m = cv2.GaussianBlur(np.roll(np.roll(mask, int(dy * N), 0), int(dx * N), 1), (0, 0), blur)
        self.put(m, (0, 0, 0), alpha)

    def done(self):
        out = np.dstack([self.rgb, self.a * 255]).clip(0, 255).astype(np.uint8)
        return cv2.resize(out, (512, 512), interpolation=cv2.INTER_AREA)


def P(x, y):
    return (int(x * N), int(y * N))


def R(r):
    return max(1, int(r * N))


def circle_mask(cx, cy, r):
    m = Canvas.mask()
    cv2.circle(m, P(cx, cy), R(r), 255, -1, cv2.LINE_AA)
    return m


def poly_mask(pts):
    m = Canvas.mask()
    cv2.fillPoly(m, [np.array([P(x, y) for x, y in pts], np.int32)], 255, cv2.LINE_AA)
    return m


def outline(c, mask, color=(40, 30, 30), width=0.018, alpha=0.9):
    k = max(3, int(width * N)) | 1
    ring = cv2.subtract(cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), mask)
    c.put(ring, color, alpha)


def highlight(c, cx, cy, rx, ry, ang=-30, alpha=0.55):
    m = Canvas.mask()
    cv2.ellipse(m, P(cx, cy), (R(rx), R(ry)), ang, 0, 360, 255, -1, cv2.LINE_AA)
    c.put(cv2.GaussianBlur(m, (0, 0), 10), (255, 255, 255), alpha)


def line_mask(pts, width):
    m = Canvas.mask()
    cv2.polylines(m, [np.array([P(x, y) for x, y in pts], np.int32)], False, 255, R(width), cv2.LINE_AA)
    return m


def star_pts(cx, cy, r, inner=0.45, n=5, rot=-90):
    pts = []
    for i in range(n * 2):
        a = math.radians(rot + i * 180 / n)
        rr = r if i % 2 == 0 else r * inner
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    return pts


def heart_pts(cx, cy, s):
    pts = []
    for t in np.linspace(0, 2 * math.pi, 120):
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((cx + x * s / 17, cy - y * s / 17))
    return pts


# =========================================================================== objets
def _heart(c, full=True):
    pts = heart_pts(0.5, 0.52, 0.42)
    m = poly_mask(pts)
    c.shadow(m)
    c.put(m, c.shade((60, 50, 235) if full else (70, 70, 80), 0.5, 0.5, 0.42))
    outline(c, m, (30, 20, 90) if full else (30, 30, 30))
    if full:
        highlight(c, 0.33, 0.33, 0.09, 0.05, -35)


def _star(c):
    pts = star_pts(0.5, 0.53, 0.44)
    m = poly_mask(pts)
    m = cv2.GaussianBlur(m, (0, 0), 3)  # coins légèrement arrondis
    c.shadow(m)
    c.put(m, c.shade((40, 200, 255), 0.5, 0.5, 0.44, strength=0.45))
    inner = poly_mask(star_pts(0.5, 0.53, 0.24))
    c.put(cv2.GaussianBlur(inner, (0, 0), 20), (150, 245, 255), 0.5)
    outline(c, m, (20, 110, 170), 0.016)
    highlight(c, 0.38, 0.33, 0.07, 0.035, -30, 0.6)


def _bomb(c):
    m = circle_mask(0.47, 0.58, 0.33)
    c.shadow(m)
    c.put(m, c.shade((70, 60, 60), 0.47, 0.58, 0.33, strength=0.7))
    cap = poly_mask([(0.58, 0.2), (0.72, 0.28), (0.66, 0.38), (0.52, 0.3)])
    c.put(cap, c.shade((110, 100, 100), 0.62, 0.29, 0.1))
    outline(c, cv2.max(m, cap), (20, 20, 20), 0.016)
    c.put(line_mask([(0.66, 0.24), (0.72, 0.15), (0.8, 0.12), (0.86, 0.15)], 0.022), (60, 120, 170))
    for k, (col, r) in enumerate((((40, 140, 255), 0.09), ((60, 220, 255), 0.055), ((230, 255, 255), 0.025))):
        sp = poly_mask(star_pts(0.88, 0.13, r, 0.4, 8, k * 11))
        c.put(cv2.GaussianBlur(sp, (0, 0), 3), col)
    highlight(c, 0.36, 0.45, 0.09, 0.05, -40, 0.45)


def _snow(c):
    g = circle_mask(0.5, 0.5, 0.42)
    c.put(cv2.GaussianBlur(g, (0, 0), 30), (255, 220, 150), 0.6)
    for k in range(6):
        a = math.radians(k * 60 - 90)
        ex, ey = 0.5 + math.cos(a) * 0.36, 0.5 + math.sin(a) * 0.36
        arm = [(0.5, 0.5), (ex, ey)]
        c.put(line_mask(arm, 0.05), (120, 60, 20))
        c.put(line_mask(arm, 0.032), (255, 245, 220))
        for t in (0.55, 0.78):
            bx, by = 0.5 + math.cos(a) * 0.36 * t, 0.5 + math.sin(a) * 0.36 * t
            for s in (-1, 1):
                b = a + s * math.radians(40)
                br = [(bx, by), (bx + math.cos(b) * 0.1, by + math.sin(b) * 0.1)]
                c.put(line_mask(br, 0.04), (120, 60, 20))
                c.put(line_mask(br, 0.024), (255, 245, 220))
    c.put(circle_mask(0.5, 0.5, 0.06), (255, 250, 235))


def _clock(c):
    m = circle_mask(0.5, 0.52, 0.4)
    c.shadow(m)
    c.put(m, c.shade((220, 120, 40), 0.5, 0.52, 0.4))
    face = circle_mask(0.5, 0.52, 0.31)
    c.put(face, c.shade((250, 250, 250), 0.5, 0.52, 0.31, strength=0.25, rim=0.1))
    for k in range(12):
        a = math.radians(k * 30)
        c.put(line_mask([(0.5 + math.cos(a) * 0.25, 0.52 + math.sin(a) * 0.25),
                         (0.5 + math.cos(a) * 0.28, 0.52 + math.sin(a) * 0.28)], 0.015), (80, 60, 50))
    c.put(line_mask([(0.5, 0.52), (0.5, 0.31)], 0.03), (60, 40, 40))
    c.put(line_mask([(0.5, 0.52), (0.64, 0.6)], 0.03), (60, 40, 40))
    c.put(circle_mask(0.5, 0.52, 0.03), (40, 40, 220))
    btn = poly_mask([(0.45, 0.06), (0.55, 0.06), (0.55, 0.13), (0.45, 0.13)])
    c.put(btn, (190, 100, 30))
    outline(c, cv2.max(m, btn), (90, 40, 10), 0.014)


def _flame(c):
    def drop(cx, cy, w, h):
        pts = []
        for t in np.linspace(0, 2 * math.pi, 100):
            x = math.sin(t) * w
            y = -math.cos(t) * h
            if y < 0:
                x *= (1 + y / h) ** 0.9  # pointe en haut
            pts.append((cx + x, cy + y * (1.35 if y < 0 else 0.7)))
        return pts
    for (col, w, h, cy) in (((30, 60, 230), 0.33, 0.36, 0.6), ((30, 140, 255), 0.24, 0.27, 0.64),
                            ((90, 230, 255), 0.13, 0.16, 0.7)):
        m = poly_mask(drop(0.5, cy, w, h))
        c.put(cv2.GaussianBlur(m, (0, 0), 2), col)
    highlight(c, 0.42, 0.56, 0.04, 0.08, 10, 0.35)


def _rock(c):
    rng = np.random.default_rng(5)
    pts = []
    for k in range(11):
        a = 2 * math.pi * k / 11
        r = 0.36 * rng.uniform(0.82, 1.08)
        pts.append((0.5 + math.cos(a) * r, 0.5 + math.sin(a) * r))
    m = poly_mask(pts)
    c.shadow(m)
    c.put(m, c.shade((70, 95, 120), 0.5, 0.5, 0.36, strength=0.75))
    for x, y, r in ((0.4, 0.42, 0.07), (0.6, 0.58, 0.09), (0.58, 0.35, 0.045), (0.38, 0.62, 0.05)):
        cm = circle_mask(x, y, r)
        c.put(cm, (45, 60, 80), 0.8)
        c.put(circle_mask(x - r * 0.2, y - r * 0.2, r * 0.75), (60, 80, 105), 0.9)
    outline(c, m, (25, 30, 40), 0.016)
    highlight(c, 0.38, 0.32, 0.08, 0.04, -30, 0.3)


# --------------------------------------------------------------------------- fruits
def _watermelon(c):
    m = Canvas.mask()
    cv2.ellipse(m, P(0.5, 0.52), (R(0.42), R(0.32)), -20, 0, 360, 255, -1, cv2.LINE_AA)
    c.shadow(m)
    c.put(m, c.shade((70, 185, 80), 0.5, 0.52, 0.4))
    # rayures sombres ondulées dans le sens de la longueur
    a = math.radians(-20)
    u = (c.xx - 0.5) * math.cos(a) + (c.yy - 0.52) * math.sin(a)
    v = -(c.xx - 0.5) * math.sin(a) + (c.yy - 0.52) * math.cos(a)
    stripes = (np.sin(v * 52 + np.sin(u * 30) * 1.2) > 0.35).astype(np.uint8) * 255
    stripes = cv2.GaussianBlur(stripes, (0, 0), 3)
    c.put(cv2.bitwise_and(stripes, m), (35, 105, 35), 0.85)
    outline(c, m, (20, 60, 20))
    highlight(c, 0.35, 0.36, 0.1, 0.04, -25, 0.4)


def _orange(c):
    m = circle_mask(0.5, 0.54, 0.38)
    c.shadow(m)
    c.put(m, c.shade((30, 140, 250), 0.5, 0.54, 0.38))
    rng = np.random.default_rng(2)
    dots = Canvas.mask()
    for _ in range(140):
        a, r = rng.uniform(0, 2 * math.pi), 0.36 * math.sqrt(rng.uniform())
        cv2.circle(dots, P(0.5 + math.cos(a) * r, 0.54 + math.sin(a) * r), R(0.006), 255, -1, cv2.LINE_AA)
    c.put(dots, (20, 110, 210), 0.5)
    leaf = Canvas.mask()
    cv2.ellipse(leaf, P(0.6, 0.15), (R(0.12), R(0.05)), -25, 0, 360, 255, -1, cv2.LINE_AA)
    c.put(leaf, c.shade((60, 170, 60), 0.6, 0.15, 0.12))
    c.put(line_mask([(0.5, 0.17), (0.52, 0.23)], 0.022), (40, 80, 90))
    outline(c, cv2.max(m, leaf), (20, 70, 140))
    highlight(c, 0.36, 0.38, 0.09, 0.05, -35, 0.5)


def _apple(c):
    m = cv2.max(circle_mask(0.4, 0.56, 0.3), circle_mask(0.6, 0.56, 0.3))
    notch = circle_mask(0.5, 0.24, 0.07)
    m = cv2.subtract(m, notch)
    bottom = circle_mask(0.5, 0.92, 0.05)
    m = cv2.subtract(m, bottom)
    c.shadow(m)
    c.put(m, c.shade((50, 40, 215), 0.5, 0.55, 0.38))
    c.put(line_mask([(0.5, 0.3), (0.52, 0.15), (0.56, 0.1)], 0.028), (40, 70, 110))
    leaf = Canvas.mask()
    cv2.ellipse(leaf, P(0.64, 0.16), (R(0.11), R(0.045)), -30, 0, 360, 255, -1, cv2.LINE_AA)
    c.put(leaf, c.shade((60, 180, 70), 0.64, 0.16, 0.11))
    outline(c, cv2.max(m, leaf), (20, 20, 100))
    highlight(c, 0.34, 0.42, 0.06, 0.1, 15, 0.5)


def _lemon(c):
    pts = []
    for t in np.linspace(0, 2 * math.pi, 120):
        x, y = math.cos(t), math.sin(t)
        tip = 1 + 0.18 * abs(math.cos(t)) ** 8
        pts.append((0.5 + x * 0.42 * tip, 0.52 + y * 0.3))
    m = poly_mask(pts)
    M = cv2.getRotationMatrix2D((N / 2, N / 2), -25, 1.0)
    m = cv2.warpAffine(m, M, (N, N))
    c.shadow(m)
    c.put(m, c.shade((40, 225, 250), 0.5, 0.52, 0.4))
    outline(c, m, (20, 120, 150))
    highlight(c, 0.38, 0.38, 0.1, 0.04, -25, 0.5)


def _kiwi(c):
    m = Canvas.mask()
    cv2.ellipse(m, P(0.5, 0.53), (R(0.4), R(0.33)), 15, 0, 360, 255, -1, cv2.LINE_AA)
    c.shadow(m)
    c.put(m, c.shade((60, 100, 140), 0.5, 0.53, 0.4, strength=0.5))
    rng = np.random.default_rng(9)
    fuzz = Canvas.mask()
    for _ in range(400):
        a, r = rng.uniform(0, 2 * math.pi), 0.38 * math.sqrt(rng.uniform())
        x, y = 0.5 + math.cos(a) * r, 0.53 + math.sin(a) * r * 0.82
        cv2.line(fuzz, P(x, y), P(x + 0.008, y + 0.01), 255, 2, cv2.LINE_AA)
    c.put(cv2.bitwise_and(fuzz, m), (40, 70, 100), 0.5)
    outline(c, m, (20, 40, 60))
    highlight(c, 0.38, 0.38, 0.09, 0.04, -20, 0.3)


def _plum(c):
    m = circle_mask(0.5, 0.54, 0.37)
    c.shadow(m)
    c.put(m, c.shade((140, 60, 200), 0.5, 0.54, 0.37))
    crease = Canvas.mask()
    cv2.ellipse(crease, P(0.56, 0.54), (R(0.06), R(0.32)), 8, -80, 80, 255, R(0.014), cv2.LINE_AA)
    c.put(cv2.bitwise_and(crease, m), (90, 30, 140), 0.7)
    c.put(line_mask([(0.5, 0.2), (0.48, 0.1)], 0.024), (40, 70, 100))
    outline(c, m, (60, 20, 90))
    highlight(c, 0.36, 0.4, 0.08, 0.05, -35, 0.5)


# --------------------------------------------------------------------------- mains (Jacques a dit)
# positions des articulations d'une main stylisée (repère 0..1, paume face à l'écran)
_FING = [(0.36, 0.42), (0.47, 0.38), (0.58, 0.4), (0.67, 0.46)]  # bases index → auriculaire
_LEN = [0.3, 0.33, 0.3, 0.24]


def _hand(c, ext, thumb="side", tilt=0.0, ox=0.0, scale=1.0, skin=(120, 200, 250)):
    """ext : 4 booléens (index → auriculaire) ; thumb : 'side' | 'up' | 'in' | 'out'"""
    def T(x, y):
        x, y = (x - 0.5) * scale, (y - 0.6) * scale
        ca, sa = math.cos(tilt), math.sin(tilt)
        return (0.5 + ox + x * ca - y * sa, 0.6 + x * sa + y * ca)
    dark = tuple(int(v * 0.55) for v in skin)
    parts = []
    palm = poly_mask([T(0.3, 0.42), T(0.72, 0.44), T(0.7, 0.78), T(0.56, 0.9), T(0.38, 0.9), T(0.28, 0.75)])
    palm = cv2.GaussianBlur(palm, (0, 0), 10)
    _, palm = cv2.threshold(palm, 120, 255, cv2.THRESH_BINARY)
    parts.append(palm)
    finger_w = 0.105 * scale
    for (bx, by), L, e in zip(_FING, _LEN, ext):
        if e:
            parts.append(line_mask([T(bx, by + 0.04), T(bx - 0.02 + (bx - 0.5) * 0.25, by - L)], finger_w))
        else:
            parts.append(line_mask([T(bx, by + 0.06), T(bx, by - 0.02)], finger_w))
    tp = {"side": [T(0.33, 0.7), T(0.16, 0.52)], "up": [T(0.33, 0.62), T(0.3, 0.2)],
          "in": [T(0.33, 0.72), T(0.52, 0.6)], "out": [T(0.33, 0.7), T(0.12, 0.6)]}[thumb]
    parts.append(line_mask(tp, finger_w * 1.08))
    m = parts[0]
    for p in parts[1:]:
        m = cv2.max(m, p)
    c.shadow(m, blur=14, alpha=0.3)
    c.put(m, c.shade(skin, 0.5, 0.55, 0.45, strength=0.45, rim=0.15))
    # plis : doigts repliés
    for (bx, by), e in zip(_FING, ext):
        if not e:
            c.put(line_mask([T(bx - 0.035, by + 0.045), T(bx + 0.035, by + 0.045)], 0.012), dark, 0.6)
    outline(c, m, tuple(int(v * 0.35) for v in skin), 0.016)


HAND_POSES = {
    "open": dict(ext=[1, 1, 1, 1], thumb="side"),
    "fist": dict(ext=[0, 0, 0, 0], thumb="in"),
    "point": dict(ext=[1, 0, 0, 0], thumb="in"),
    "peace": dict(ext=[1, 1, 0, 0], thumb="in"),
    "rock": dict(ext=[1, 0, 0, 1], thumb="out"),
    "thumbs_up": dict(ext=[0, 0, 0, 0], thumb="up"),
}


def _thumbs_up(c, skin=(120, 200, 250)):
    """Poing de profil, pouce dressé."""
    fist = Canvas.mask()
    x0, y0, x1, y1 = 0.3, 0.48, 0.74, 0.9
    r = R(0.08)
    cv2.rectangle(fist, (R(x0) + r, R(y0)), (R(x1) - r, R(y1)), 255, -1)
    cv2.rectangle(fist, (R(x0), R(y0) + r), (R(x1), R(y1) - r), 255, -1)
    for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        cv2.circle(fist, (R(cx) + (r if cx == x0 else -r), R(cy) + (r if cy == y0 else -r)), r, 255, -1, cv2.LINE_AA)
    thumb = line_mask([(0.39, 0.55), (0.41, 0.2)], 0.15)
    m = cv2.max(fist, thumb)
    c.shadow(m, blur=14, alpha=0.3)
    c.put(m, c.shade(skin, 0.5, 0.55, 0.45, strength=0.45, rim=0.15))
    dark = tuple(int(v * 0.55) for v in skin)
    for k in range(4):  # doigts repliés vus de côté
        y = 0.56 + k * 0.085
        c.put(line_mask([(0.56, y), (0.74, y)], 0.012), dark, 0.6)
    c.put(line_mask([(0.46, 0.5), (0.46, 0.88)], 0.012), dark, 0.4)
    outline(c, m, tuple(int(v * 0.35) for v in skin), 0.016)


def _hand_pose(name):
    def draw(c):
        if name == "thumbs_up":
            _thumbs_up(c)
        else:
            _hand(c, **HAND_POSES[name])
    return draw


def _clap(c):
    _hand(c, [1, 1, 1, 1], "side", tilt=0.45, ox=-0.12, scale=0.78)
    _hand(c, [1, 1, 1, 1], "side", tilt=-0.45, ox=0.12, scale=0.78)
    for k, (x, y, a) in enumerate(((0.2, 0.2, 225), (0.5, 0.08, 270), (0.8, 0.2, 315))):
        ar = math.radians(a)
        c.put(line_mask([(x, y), (x + math.cos(ar) * 0.06, y + math.sin(ar) * 0.06)], 0.025), (60, 200, 255))


# --------------------------------------------------------------------------- icônes de l'arcade
def _app_bg(c, top, bottom):
    m = Canvas.mask()
    r = R(0.2)
    x0, y0, x1, y1 = R(0.04), R(0.04), R(0.96), R(0.96)
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    c.shadow(m, 0, 0.015, 20, 0.4)
    k = c.yy[..., None]
    grad = np.asarray(top, np.float32) * (1 - k) + np.asarray(bottom, np.float32) * k
    c.put(m, grad)
    gloss = Canvas.mask()
    cv2.ellipse(gloss, P(0.5, 0.05), (R(0.62), R(0.34)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    c.put(cv2.bitwise_and(cv2.GaussianBlur(gloss, (0, 0), 25), m), (255, 255, 255), 0.18)
    return m


def _paste_emoji(c, char, cx, cy, size, rot=0.0, alpha=1.0):
    """Colle un emoji Apple (iOS) sur la toile, en haute qualité, avec une ombre douce."""
    import ui
    rgba = ui.emoji_rgba(char)
    s = R(size)
    hh, ww = rgba.shape[:2]
    sc = s / max(hh, ww)
    rgb, al = ui.resize_rgba(rgba, (max(1, int(ww * sc)), max(1, int(hh * sc))))
    if rot:
        M = cv2.getRotationMatrix2D((rgb.shape[1] / 2, rgb.shape[0] / 2), rot, 1.0)
        rgb = cv2.warpAffine(rgb.astype(np.float32), M, (rgb.shape[1], rgb.shape[0]))
        al = cv2.warpAffine(al.astype(np.float32), M, (al.shape[1], al.shape[0]))[..., None]
    h, w = al.shape[:2]
    x0, y0 = int(cx * N - w / 2), int(cy * N - h / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(N, x0 + w), min(N, y0 + h)
    if xa >= xb or ya >= yb:
        return
    a = al[ya - y0:yb - y0, xa - x0:xb - x0] * alpha
    sh = np.zeros((N, N), np.uint8)
    sh[ya:yb, xa:xb] = (a[..., 0] * 255).astype(np.uint8)
    c.shadow(sh, 0.0, 0.018, 16, 0.35)
    col = rgb[ya - y0:yb - y0, xa - x0:xb - x0][..., ::-1]  # RGB → BGR
    c.rgb[ya:yb, xa:xb] = c.rgb[ya:yb, xa:xb] * (1 - a) + col * a
    c.a[ya:yb, xa:xb] = c.a[ya:yb, xa:xb] + a[..., 0] * (1 - c.a[ya:yb, xa:xb])


def _sparkles(c, pts):
    for x, y, r in pts:
        c.put(cv2.GaussianBlur(poly_mask(star_pts(x, y, r, 0.3, 4)), (0, 0), 2), (255, 255, 255))


def _icon_danse(c):
    _app_bg(c, (210, 70, 170), (110, 25, 90))
    _paste_emoji(c, "🪩", 0.74, 0.27, 0.3)
    _paste_emoji(c, "💃", 0.44, 0.56, 0.62)
    _paste_emoji(c, "🎵", 0.8, 0.7, 0.2, rot=-10)
    _sparkles(c, [(0.2, 0.22, 0.035), (0.86, 0.48, 0.025)])


def _icon_bulles(c):
    _app_bg(c, (245, 195, 70), (175, 95, 20))
    _paste_emoji(c, "🫧", 0.5, 0.52, 0.72)
    _sparkles(c, [(0.22, 0.24, 0.035), (0.82, 0.3, 0.03), (0.8, 0.8, 0.025)])


def _icon_tranche(c):
    _app_bg(c, (100, 205, 80), (40, 115, 25))
    _paste_emoji(c, "🍉", 0.5, 0.54, 0.66, rot=-12)
    swoosh = Canvas.mask()
    cv2.ellipse(swoosh, P(0.5, 0.54), (R(0.44), R(0.08)), -40, 180, 360, 255, R(0.03), cv2.LINE_AA)
    c.put(cv2.GaussianBlur(swoosh, (0, 0), 10), (255, 255, 210), 0.7)
    c.put(swoosh, (255, 255, 255))


def _icon_jacques(c):
    _app_bg(c, (70, 160, 255), (30, 75, 205))
    bubble = Canvas.mask()
    cv2.ellipse(bubble, P(0.5, 0.42), (R(0.37), R(0.28)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    bubble = cv2.max(bubble, poly_mask([(0.3, 0.6), (0.22, 0.82), (0.46, 0.66)]))
    c.shadow(bubble, 0.01, 0.015, 12, 0.3)
    c.put(bubble, (255, 255, 255))
    _paste_emoji(c, "✌️", 0.5, 0.42, 0.4)
    _paste_emoji(c, "🙋", 0.76, 0.76, 0.3)


def _icon_esquive(c):
    m = _app_bg(c, (95, 45, 30), (40, 15, 10))
    rng = np.random.default_rng(3)
    st = Canvas.mask()
    for _ in range(45):
        cv2.circle(st, P(rng.uniform(0.08, 0.92), rng.uniform(0.08, 0.92)), R(rng.uniform(0.003, 0.008)), 255, -1,
                   cv2.LINE_AA)
    c.put(cv2.bitwise_and(st, m), (255, 255, 255), 0.9)
    _paste_emoji(c, "☄️", 0.58, 0.42, 0.6)
    _paste_emoji(c, "😮", 0.26, 0.78, 0.24)


DRAW = {
    "heart": lambda c: _heart(c, True), "heart_empty": lambda c: _heart(c, False), "star": _star, "bomb": _bomb,
    "snow": _snow, "clock": _clock, "flame": _flame, "rock": _rock,
    "watermelon": _watermelon, "orange": _orange, "apple": _apple, "lemon": _lemon, "kiwi": _kiwi, "plum": _plum,
    "clap": _clap,
    "icon_danse": _icon_danse, "icon_bulles": _icon_bulles, "icon_tranche": _icon_tranche,
    "icon_jacques": _icon_jacques, "icon_esquive": _icon_esquive,
}
for _k in HAND_POSES:
    DRAW["hand_" + _k] = _hand_pose(_k)

# couleur du jus des fruits (BGR)
JUICE = {"watermelon": (70, 60, 235), "orange": (30, 150, 255), "apple": (90, 220, 240), "lemon": (60, 235, 250),
         "kiwi": (60, 200, 120), "plum": (160, 60, 200)}


def base(name):
    if name not in _base:
        c = Canvas()
        DRAW[name](c)
        _base[name] = c.done()
    return _base[name]


def sprite(name, size):
    """Icône → (bgr float32, alpha float32) au format de ui.blit."""
    size = max(8, int(size) // 2 * 2)
    key = (name, size)
    if key not in _sized:
        if len(_sized) > 600:
            _sized.clear()
        im = cv2.resize(base(name), (size, size), interpolation=cv2.INTER_AREA)
        a = im.astype(np.float32) / 255.0
        _sized[key] = (a[..., :3] * 255.0, a[..., 3:4])
    return _sized[key]


def prewarm(names=None):
    import ui
    for n in names or APP_ICONS:
        base(n)
    for ch in NAME_TO_EMOJI.values():
        ui.emoji_rgba(ch)


# noms d'objets → emoji Apple (iOS)
NAME_TO_EMOJI = {"star": "⭐", "bomb": "💣", "snow": "❄️", "clock": "⏱️", "flame": "🔥", "heart": "❤️",
                 "heart_empty": "🖤", "rock": "🪨", "clap": "👏", "watermelon": "🍉", "orange": "🍊", "apple": "🍎",
                 "lemon": "🍋", "kiwi": "🥝", "plum": "🍑", "hand_open": "✋", "hand_fist": "✊",
                 "hand_point": "☝️", "hand_peace": "✌️", "hand_rock": "🤘", "hand_thumbs_up": "👍"}


def get(key, size):
    """Icônes d'app de l'arcade : dessinées ici. Tout le reste : emojis Apple (iOS) haute qualité."""
    import ui
    if key.startswith("icon_"):
        return sprite(key, size)
    return ui.emoji(NAME_TO_EMOJI.get(key, key), size)
