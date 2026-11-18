"""Effets appliqués dans le cadre formé par les doigts.

Chaque effet : fn(img, ctx) -> image de même taille. `ctx` contient :
  cell   taille de base réglée avec [ ]  (en "pixels 720p", multipliée par l'échelle)
  t, dt  temps écoulé / durée de la frame
  color  mode couleur (ascii / matrix)
  quad   cadre (haut-gauche, haut-droite, bas-droite, bas-gauche) ou None
  center centre du cadre
  prev   image finale de la frame précédente (pour le tunnel infini)
  history  mémoire des images passées (portail, clones, rgb-temps)
  mask   silhouette (0..1, pleine résolution) si l'effet la demande
  bg     image du fond sans toi (cape d'invisibilité)
  state  dict persistant pour les effets qui ont une mémoire

Pour ajouter un effet : une fonction + une ligne dans EFFECTS (tout en bas).
"""
import math
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MONO_FONT = "/System/Library/Fonts/Menlo.ttc"
KANA_FONTS = ["/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/AquaKana.ttc"]
COLOR_MODES = ["couleur", "vert", "ambre", "cyan", "blanc"]
_cache = {}


def _scale(img):
    return img.shape[0] / 720.0


# =========================================================================== moteur de glyphes
def _font(path_list, size):
    for p in ([path_list] if isinstance(path_list, str) else path_list):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _glyphs(chars, cw, ch, fonts, size, mirror=False):
    key = ("g", chars, cw, ch, size, mirror)
    if key not in _cache:
        f = _font(fonts, size)
        tiles = np.zeros((len(chars), ch, cw), np.float32)
        for i, c in enumerate(chars):
            im = Image.new("L", (cw, ch), 0)
            ImageDraw.Draw(im).text((cw / 2, ch / 2), c, font=f, fill=255, anchor="mm")
            a = np.asarray(im, np.float32) / 255.0
            tiles[i] = a[:, ::-1] if mirror else a
        _cache[key] = tiles
    return _cache[key]


_RAMP_CANDIDATES = " .'`^\",:;Il!i><~+?][}{1)(tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
_EDGE_CHARS = "|/-\\"


def _ascii_set(cw, ch):
    """Rampe de caractères triés par densité réelle (mesurée sur la police) + caractères d'arêtes."""
    key = ("ascii", cw, ch)
    if key not in _cache:
        size = max(6, int(round(cw / 0.6)))
        cand = _glyphs(_RAMP_CANDIDATES, cw, ch, MONO_FONT, size)
        cov = cand.mean(axis=(1, 2))
        n = 18
        used, pick = set(), []
        for tv in np.linspace(0, cov.max(), n):
            for j in np.argsort(np.abs(cov - tv)):
                if j not in used:
                    used.add(j)
                    pick.append(j)
                    break
        pick.sort(key=lambda j: cov[j])
        ramp = cand[pick]
        ramp = ramp / max(1e-6, ramp.max())
        edges = np.clip(_glyphs(_EDGE_CHARS, cw, ch, MONO_FONT, size) * 1.4, 0, 1)
        _cache[key] = (np.concatenate([ramp, edges]), len(pick))
    return _cache[key]


def _u8(tiles):
    k = ("u8", id(tiles))
    if k not in _cache:
        _cache[k] = (np.clip(tiles, 0, 1) * 255).astype(np.uint8)
    return _cache[k]


from concurrent.futures import ThreadPoolExecutor

_POOL = ThreadPoolExecutor(max_workers=8)  # NumPy / OpenCV libèrent le GIL → vrai parallélisme sur Apple Silicon


def _shrink(img, size):
    """Réduction rapide et propre : pyrDown (÷2, très rapide) tant que c'est possible, puis INTER_AREA.
    (INTER_AREA seul d'une image 4K vers une petite grille est très lent.)"""
    w, h = size
    while img.shape[1] >= 4 * w and img.shape[0] >= 4 * h:
        img = cv2.pyrDown(img)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _compose(idx, tiles, colors, bg=None, full=None, fill=0):
    """idx (rows, cols) → image de glyphes colorés.
    Calcul découpé en bandes horizontales traitées en parallèle, écrit directement dans l'image finale
    (taille `full`, marges remplies avec `fill`)."""
    rows, cols = idx.shape
    _, ch, cw = tiles.shape
    t8 = _u8(tiles)
    H, W = (full[0], full[1]) if full is not None else (rows * ch, cols * cw)
    out = np.empty((H, W, 3), np.uint8)
    if H > rows * ch:
        out[rows * ch:] = fill
    if W > cols * cw:
        out[:rows * ch, cols * cw:] = fill
    colors8 = np.clip(colors, 0, 255).astype(np.uint8)
    bg8 = np.clip(bg, 0, 255).astype(np.uint8) if bg is not None else None

    def band(r0, r1):
        n = r1 - r0
        size = (cols * cw, n * ch)
        m = t8[idx[r0:r1]].transpose(0, 2, 1, 3).reshape(n * ch, cols * cw)
        m3 = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        o = cv2.multiply(cv2.resize(colors8[r0:r1], size, interpolation=cv2.INTER_NEAREST), m3, scale=1 / 255)
        if bg8 is not None:
            b = cv2.resize(bg8[r0:r1], size, interpolation=cv2.INTER_NEAREST)
            o = cv2.add(o, cv2.multiply(b, cv2.bitwise_not(m3), scale=1 / 255))
        out[r0 * ch:r1 * ch, :cols * cw] = o

    step = max(1, -(-rows // 8))
    list(_POOL.map(lambda r: band(r, min(rows, r + step)), range(0, rows, step)))
    return out


def _place(img, block, fill=0):
    if block.shape == img.shape:
        return block
    out = np.empty_like(img)
    out[:] = fill
    h, w = block.shape[:2]
    out[:h, :w] = block
    return out


def _clahe(clip=2.5, grid=(6, 6)):
    k = ("clahe", clip, grid)
    if k not in _cache:
        _cache[k] = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return _cache[k]


def _mode_color(mode, small, lum):
    """Couleur par cellule selon le mode choisi (touche C)."""
    if mode == "couleur":
        s = small.astype(np.float32)
        vivid = s * (255.0 / (s.max(axis=2, keepdims=True) + 1.0))  # teinte pure, pleine luminosité
        return vivid * 0.65 + 255 * lum[..., None] * 0.35 + 25
    tint = {"vert": (90, 255, 120), "ambre": (40, 175, 255), "cyan": (255, 225, 90),
            "blanc": (235, 235, 235)}[mode]
    return np.array(tint, np.float32) * (0.35 + 0.65 * lum[..., None])


def _grid(sw, sh):
    key = ("grid", sw, sh)
    if key not in _cache:
        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        _cache[key] = (xx, yy)
    return _cache[key]


# =========================================================================== effets
def pixel(img, c):
    h, w = img.shape[:2]
    k = max(3, int(round(c["cell"] * _scale(img))))
    sw, sh = max(1, w // k), max(1, h // k)
    small = _shrink(img, (sw, sh))
    big = cv2.resize(small, (sw * k, sh * k), interpolation=cv2.INTER_NEAREST)
    if big.shape != img.shape:
        big = cv2.copyMakeBorder(big, 0, h - sh * k, 0, w - sw * k, cv2.BORDER_CONSTANT, value=0)
    if k >= 10:  # fine grille entre les gros pixels, look écran LED (masque mis en cache)
        key = ("pixgrid", k, w, h)
        if key not in _cache:
            g = np.full((h, w), 255, np.uint8)
            g[::k] = 150
            g[:, ::k] = 150
            _cache[key] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        big = cv2.multiply(big, _cache[key], scale=1 / 255)
    return big


def ascii_art(img, c):
    """ASCII haute qualité : densité mesurée sur la vraie police (Menlo), contraste local,
    et caractères d'arêtes ( | / - \\ ) orientés selon les contours."""
    h, w = img.shape[:2]
    cw = max(4, int(round(c["cell"] * 0.5 * _scale(img))))
    ch = int(round(cw * 1.85))
    cols, rows = max(1, w // cw), max(1, h // ch)
    tiles, n = _ascii_set(cw, ch)

    small = _shrink(img, (cols, rows))
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    g = _clahe(1.4, (max(1, cols // 32), max(1, rows // 18))).apply(g)
    lo, hi = np.percentile(g, (2, 98))
    lum = np.clip((g.astype(np.float32) - lo) / max(1.0, hi - lo), 0, 1) ** 1.25
    idx = np.clip((lum * (n - 1) + 0.5).astype(np.int32), 0, n - 1)

    # contours : tenseur de structure calculé sur une grille 4x plus fine
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g4 = _shrink(gray, (cols * 4, rows * 4)).astype(np.float32)
    g4 = cv2.GaussianBlur(g4, (0, 0), 1.0)
    gx = cv2.Sobel(g4, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g4, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.resize(gx * gx, (cols, rows), interpolation=cv2.INTER_AREA)
    jyy = cv2.resize(gy * gy, (cols, rows), interpolation=cv2.INTER_AREA)
    jxy = cv2.resize(gx * gy, (cols, rows), interpolation=cv2.INTER_AREA)
    energy = jxx + jyy
    a, b = jxx - jyy, 2 * jxy
    coh = np.sqrt(a * a + b * b) / (energy + 1e-3)
    mag = np.sqrt(energy)
    edge = (mag > max(60.0, float(np.percentile(mag, 90)))) & (coh > 0.62)
    deg = np.degrees(0.5 * np.arctan2(b, a))           # orientation du gradient
    eidx = np.where(np.abs(deg) < 22.5, 0,               # gradient horizontal → arête |
                    np.where(np.abs(deg) > 67.5, 2,      # gradient vertical → arête -
                             np.where(deg > 0, 1, 3)))   # diagonales / et \
    if c.get("ascii_edges", True):
        idx = np.where(edge, n + eidx, idx)

    col = _mode_color(c.get("color", "couleur"), small, lum)
    col = np.where(edge[..., None], col * 0.7 + 80, col)
    return _compose(idx, tiles, col, small.astype(np.float32) * 0.06, full=img.shape)


_KANA = ("アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
         "0123456789Z:・=*+<>")


def matrix(img, c):
    """Pluie de code : colonnes de caractères qui tombent, ton visage apparaît dans la lumière."""
    h, w = img.shape[:2]
    cw = max(6, int(round(c["cell"] * 0.75 * _scale(img))))
    ch = int(round(cw * 1.3))
    cols, rows = max(1, w // cw), max(1, h // ch)
    tiles = _glyphs(_KANA, cw, ch, KANA_FONTS, int(cw * 1.05), mirror=True)
    st = c["state"].setdefault("matrix", {})
    rng = np.random
    if st.get("shape") != (rows, cols):
        st.update(shape=(rows, cols), head=rng.uniform(-rows, rows, cols), speed=rng.uniform(8, 26, cols),
                  trail=rng.uniform(8, 30, cols), glyph=rng.randint(0, len(_KANA), (rows, cols)))
    st["head"] += st["speed"] * min(c.get("dt", 0.033), 0.1)
    done = st["head"] - st["trail"] > rows
    k = int(done.sum())
    if k:
        st["head"][done] = rng.uniform(-12, 0, k)
        st["speed"][done] = rng.uniform(8, 26, k)
        st["trail"][done] = rng.uniform(8, 30, k)
    flick = rng.rand(rows, cols) < 0.03
    st["glyph"][flick] = rng.randint(0, len(_KANA), int(flick.sum()))

    small = _shrink(img, (cols, rows))
    lum = _clahe(2.0, (4, 4)).apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 255.0
    d = st["head"][None, :] - np.arange(rows)[:, None]
    tr = st["trail"][None, :]
    trail = np.where((d >= 0) & (d <= tr), np.clip(1 - d / tr, 0, 1) ** 1.6, 0)
    v = np.clip(lum ** 1.15 * 1.05 + trail * 0.6, 0, 1)
    mode = c.get("color", "vert")
    col = _mode_color("vert" if mode == "couleur" else mode, small, v) * (0.25 + 1.1 * v[..., None])
    col[(d >= 0) & (d < 1.0)] = (235, 255, 235)  # tête de colonne blanche
    out = _compose(st["glyph"], tiles, col, full=img.shape)
    glow = cv2.GaussianBlur(_shrink(out, (w // 4, h // 4)), (0, 0), 3)
    return cv2.addWeighted(out, 1.0, cv2.resize(glow, (w, h)), 1.6, 0)


def tunnel(img, c):
    """Tunnel infini : toute l'image est réinjectée dans le cadre à chaque frame (effet Droste),
    avec une petite rotation → spirale."""
    q, prev = c.get("quad"), c.get("prev")
    if q is None or prev is None or prev.shape != img.shape:
        return img
    h, w = img.shape[:2]
    cx, cy = q.mean(axis=0)
    a = math.radians(2.5)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    qr = ((q - (cx, cy)) @ rot.T + (cx, cy)).astype(np.float32)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), qr)
    out = cv2.warpPerspective(prev, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.convertScaleAbs(out, alpha=0.97, beta=2)


def portal(img, c):
    """Portail temporel : dans le cadre, tu vois ce qui se passait il y a 2 secondes."""
    hist = c.get("history")
    old = hist.at(2.0)[0] if hist else None
    if old is None:
        return img
    out = cv2.convertScaleAbs(old, alpha=0.92)
    out[..., 0] = np.minimum(out[..., 0].astype(np.int16) + 18, 255).astype(np.uint8)  # teinte froide
    return out


def invisible(img, c):
    """Cape d'invisibilité façon Predator : ta silhouette laisse voir le fond, avec réfraction sur les bords."""
    bg, m = c.get("bg"), c.get("mask")
    if bg is None or m is None:
        return img
    h, w = img.shape[:2]
    sw, sh = w // 3, h // 3
    ms = cv2.GaussianBlur(cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA), (0, 0), 3)
    k = 28.0 * _scale(img) / 3
    dx = cv2.Sobel(ms, cv2.CV_32F, 1, 0, ksize=5) * k
    dy = cv2.Sobel(ms, cv2.CV_32F, 0, 1, ksize=5) * k
    xx, yy = _grid(sw, sh)
    wob = np.sin(yy * 0.15 + c["t"] * 6).astype(np.float32) * 1.5 * ms
    mapx = cv2.resize((xx + dx + wob) * 3, (w, h))
    mapy = cv2.resize((yy + dy) * 3, (w, h))
    refr = cv2.remap(bg, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = cv2.blendLinear(img, refr, 1.0 - m, m)
    edge = (np.clip(np.abs(dx) + np.abs(dy), 0, 12) / 12 * ms * 80).astype(np.uint8)
    edge = cv2.resize(edge, (w, h))
    return cv2.add(out, cv2.merge([edge, cv2.convertScaleAbs(edge, alpha=0.9), cv2.convertScaleAbs(edge, alpha=0.7)]))


def clones(img, c):
    """Clones : des copies de toi décalées dans le temps te suivent."""
    hist, m = c.get("history"), c.get("mask")
    if hist is None or m is None:
        return img
    h, w = img.shape[:2]
    out = img
    tints = [(1.0, 0.75, 1.15), (1.15, 1.0, 0.75), (0.8, 1.15, 1.0)]
    for (delay, alpha), tint in zip(((1.5, 0.55), (1.0, 0.7), (0.5, 0.85)), tints):
        f, fm = hist.at(delay)
        if f is None or fm is None:
            continue
        a = cv2.resize(fm, (w, h)) * alpha
        tinted = cv2.transform(f, np.diag(tint).astype(np.float32))
        out = cv2.blendLinear(out, tinted, 1.0 - a, a)
    return cv2.blendLinear(out, img, 1.0 - m, m)


def rgb_time(img, c):
    """Les canaux rouge / vert / bleu vivent à des instants différents → traînées arc-en-ciel."""
    hist = c.get("history")
    g = hist.at(0.15)[0] if hist else None
    b = hist.at(0.3)[0] if hist else None
    if g is None or b is None:
        return img
    return cv2.merge([b[..., 0], g[..., 1], img[..., 2]])


def thermal(img, c):
    """Caméra thermique : avec la silhouette, ton corps est « chaud » et le décor froid."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = c.get("mask")
    if m is not None:
        ms = cv2.resize(m, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        small = small * (0.35 + 0.45 * ms) + 130 * ms
    small = cv2.GaussianBlur(small, (0, 0), 2.2)
    small = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_INFERNO)


def neon(img, c):
    h, w = img.shape[:2]
    sw, sh = w // 2, h // 2
    g = cv2.cvtColor(cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    e = cv2.dilate(cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 110), None)
    key = ("hue", sw, sh)
    if key not in _cache:
        yy, xx = np.mgrid[0:sh, 0:sw]
        _cache[key] = ((xx + yy) / (sw + sh) * 120).astype(np.float32)
    hue = ((_cache[key] + c["t"] * 50) % 180).astype(np.uint8)
    col = cv2.cvtColor(cv2.merge([hue, np.full_like(e, 230), e]), cv2.COLOR_HSV2BGR)
    out = cv2.addWeighted(col, 1.0, cv2.GaussianBlur(col, (0, 0), 6), 2.4, 0)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def glitch(img, c):
    h, w = img.shape[:2]
    t, S = c["t"], _scale(img)
    k = int((10 + 8 * math.sin(t * 7)) * S)
    b, g, r = cv2.split(img)
    out = cv2.merge([np.roll(b, -k, axis=1), g, np.roll(r, k, axis=1)])
    rng = np.random.default_rng(int(t * 14))
    for _ in range(int(rng.integers(3, 9))):
        y = int(rng.integers(0, h))
        hh = int(rng.integers(4, 45) * S)
        out[y:y + hh] = np.roll(out[y:y + hh], int(rng.integers(-80, 80) * S), axis=1)
    out[::3] = (out[::3] * 0.7).astype(np.uint8)
    return out


def kaleido(img, c):
    """Kaléidoscope à 8 branches centré sur le cadre, qui tourne lentement."""
    h, w = img.shape[:2]
    cx, cy = c["center"] if c.get("center") is not None else (w / 2, h / 2)
    xx, yy = _grid(w // 2, h // 2)
    dx, dy = xx * 2 - cx, yy * 2 - cy
    r = np.sqrt(dx * dx + dy * dy)
    seg = 2 * math.pi / 8
    rot = c["t"] * 0.35
    a = np.mod(np.arctan2(dy, dx) + rot, seg)
    a = np.where(a > seg / 2, seg - a, a) - rot * 0.5
    mapx = cv2.resize((cx + r * np.cos(a)).astype(np.float32), (w, h))
    mapy = cv2.resize((cy + r * np.sin(a)).astype(np.float32), (w, h))
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def bubble(img, c):
    """Bulle / loupe fisheye : le centre du cadre est grossi, les bords se raccordent ([ ] = force)."""
    q = c.get("quad")
    if q is None:
        return img
    h, w = img.shape[:2]
    cx, cy = q.mean(axis=0)
    R = float(np.max(np.linalg.norm(q - (cx, cy), axis=1))) + 1
    xx, yy = _grid(w // 2, h // 2)
    dx, dy = xx * 2 - cx, yy * 2 - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    k = 1.0 + min(max(c["cell"] / 14.0, 0.4), 3.0)
    f = np.where(r < R, np.clip(r / R, 0, 1) ** (k - 1), 1.0)
    mapx = cv2.resize((cx + dx * f).astype(np.float32), (w, h))
    mapy = cv2.resize((cy + dy * f).astype(np.float32), (w, h))
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def sketch(img, c):
    """Croquis au crayon sur papier."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(255 - gray, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    blur = cv2.resize(cv2.GaussianBlur(small, (0, 0), 7 * _scale(img) / 3), (w, h))
    s = cv2.divide(gray, 255 - blur, scale=256)
    key = ("paper", w, h)
    if key not in _cache:
        n = np.random.default_rng(1).normal(0, 1, (h // 2, w // 2)).astype(np.float32)
        _cache[key] = cv2.resize(cv2.GaussianBlur(n, (0, 0), 1.2), (w, h)) * 6
    sf = cv2.add(s.astype(np.float32) * 0.97, _cache[key])
    return cv2.transform(cv2.merge([sf, sf, sf]), np.diag([0.88, 0.95, 1.0]).astype(np.float32)).clip(0, 255).astype(np.uint8)


def halftone(img, c):
    h, w = img.shape[:2]
    k = max(5, int(round(c["cell"] * _scale(img))))
    cols, rows = max(1, w // k), max(1, h // k)
    small = _shrink(img, (cols, rows))
    dark = 255 - cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    key = ("dots", k)
    if key not in _cache:
        lv, ss = 16, 4
        tiles = np.zeros((lv, k, k), np.float32)
        for i in range(lv):
            big = np.zeros((k * ss, k * ss), np.uint8)
            r = math.sqrt(i / (lv - 1)) * k * ss * 0.62
            if r > 0.5:
                cv2.circle(big, (k * ss // 2, k * ss // 2), int(r), 255, -1, cv2.LINE_AA)
            tiles[i] = cv2.resize(big, (k, k), interpolation=cv2.INTER_AREA) / 255.0
        _cache[key] = tiles
    tiles = _cache[key]
    idx = np.clip((dark.astype(np.int32) * len(tiles)) // 256, 0, len(tiles) - 1)
    paper, ink = (215, 235, 242), (60, 30, 25)
    ink_c = np.empty((rows, cols, 3), np.float32)
    ink_c[:] = ink
    paper_c = np.empty((rows, cols, 3), np.float32)
    paper_c[:] = paper
    return _compose(idx, tiles, ink_c, paper_c, full=img.shape, fill=paper)


def night(img, c):
    h, w = img.shape[:2]
    g = _clahe(3.0, (8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    noise = np.random.randint(0, 60, (h // 2, w // 2), dtype=np.uint8)
    g += cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32) - 30
    key = ("vig", w, h)
    if key not in _cache:
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        _cache[key] = np.clip(1.25 - 0.6 * d, 0, 1).astype(np.float32)
    g *= _cache[key]
    out = cv2.merge([g * 0.25, g * 1.05, g * 0.35])
    out[::2] *= 0.85
    return np.clip(out, 0, 255).astype(np.uint8)


def invert(img, c):
    return 255 - img


def _soft_blur(img, sigma):
    h, w = img.shape[:2]
    f = 1
    while sigma / (f * 2) >= 2.5 and min(h, w) // (f * 2) > 16:
        f *= 2
    if f == 1:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    small = cv2.GaussianBlur(cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA), (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def frosted(img, c):
    """Flou / verre dépoli : flou doux et lisse, léger voile et grain ([ ] = intensité)."""
    h, w = img.shape[:2]
    out = _soft_blur(img, max(2.0, c["cell"] * 1.4 * _scale(img)))
    key = ("grain", w, h)
    if key not in _cache:
        _cache[key] = np.random.default_rng(3).integers(0, 14, (h, w, 1), dtype=np.uint8).repeat(3, axis=2)
    out = cv2.addWeighted(out, 0.9, np.full_like(out, 255), 0.1, 0)
    return cv2.add(out, _cache[key])


def aura(img, c):
    """Aura : un halo néon arc-en-ciel épouse ta silhouette."""
    m = c.get("mask")
    if m is None:
        return img
    h, w = img.shape[:2]
    sw, sh = w // 4, h // 4
    ms = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
    edge = np.clip(cv2.dilate(ms, np.ones((5, 5), np.uint8)) - cv2.erode(ms, np.ones((3, 3), np.uint8)), 0, 1)
    key = ("hue", sw, sh)
    if key not in _cache:
        yy, xx = np.mgrid[0:sh, 0:sw]
        _cache[key] = ((xx + yy) / (sw + sh) * 120).astype(np.float32)
    hue = ((_cache[key] * 1.5 + c["t"] * 70) % 180).astype(np.uint8)
    col = cv2.cvtColor(cv2.merge([hue, np.full_like(hue, 255), (edge * 255).astype(np.uint8)]), cv2.COLOR_HSV2BGR)
    glow = cv2.addWeighted(_soft_blur(col, 6), 2.5, _soft_blur(col, 2), 1.5, 0)
    dark = cv2.convertScaleAbs(img, alpha=0.55)
    base = cv2.blendLinear(dark, img, 1.0 - m, m)
    return cv2.add(base, cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR))


def space(img, c):
    """Espace : ton décor devient un champ d'étoiles en hyper-vitesse, toi tu restes."""
    m = c.get("mask")
    h, w = img.shape[:2]
    st = c["state"].setdefault("space", {})
    n = 700
    if st.get("size") != (w, h):
        rng = np.random.default_rng()
        st.update(size=(w, h), xy=rng.uniform(-1, 1, (n, 2)).astype(np.float32),
                  z=rng.uniform(0.05, 1, n).astype(np.float32))
    z = st["z"]
    z -= min(c.get("dt", 0.033), 0.1) * 0.35
    dead = z < 0.03
    if dead.any():
        st["xy"][dead] = np.random.uniform(-1, 1, (int(dead.sum()), 2))
        z[dead] = 1.0
    cx, cy = c["center"] if c.get("center") is not None else (w / 2, h / 2)
    p = st["xy"] / z[:, None] * (0.5 * w)
    p2 = st["xy"] / (z[:, None] + 0.04) * (0.5 * w)
    out = np.zeros_like(img)
    key = ("nebula", w, h)
    if key not in _cache:
        rng = np.random.default_rng(7)
        neb = rng.uniform(0, 1, (h // 32, w // 32, 3)).astype(np.float32)
        neb = cv2.resize(neb, (w, h), interpolation=cv2.INTER_CUBIC)
        _cache[key] = np.clip(neb * np.array([90, 30, 70], np.float32), 0, 255).astype(np.uint8)
    out[:] = _cache[key]
    for (x1, y1), (x0, y0), zz in zip(p, p2, z):
        b = int(255 * min(1.0, (1 - zz) * 1.3))
        cv2.line(out, (int(cx + x0), int(cy + y0)), (int(cx + x1), int(cy + y1)), (b, b, b),
                 max(1, int((1 - zz) * 3 * _scale(img))), cv2.LINE_AA)
    if m is None:
        return out
    return cv2.blendLinear(out, img, 1.0 - m, m)


def mirror(img, c):
    """Miroir : la moitié gauche du cadre se reflète à droite (visages symétriques garantis)."""
    h, w = img.shape[:2]
    cx = int(c["center"][0]) if c.get("center") is not None else w // 2
    out = img.copy()
    left = img[:, max(0, 2 * cx - w):cx]
    out[:, cx:cx + left.shape[1]] = left[:, ::-1]
    return out


def vhs(img, c):
    """Cassette VHS : couleurs qui bavent, bruit, lignes de tracking, ondulation, date incrustée."""
    h, w = img.shape[:2]
    t, S = c["t"], _scale(img)
    small = cv2.resize(img, (w // 3, h // 2), interpolation=cv2.INTER_AREA)
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    cr = np.roll(cv2.blur(cr, (9, 1)), int(3 * S), axis=1)
    cb = np.roll(cv2.blur(cb, (9, 1)), int(-2 * S), axis=1)
    out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    rows = np.arange(h, dtype=np.float32)
    shift = (np.sin(rows * 0.02 + t * 3) * 2 * S + np.where(np.abs((rows / h - (t * 0.25) % 1.2)) < 0.03,
                                                             18 * S, 0)).astype(np.int32)
    xx, _ = _grid(w, 1)
    mapx = (xx[0][None, :] - shift[:, None]).astype(np.float32)
    mapy = np.repeat(rows[:, None], w, axis=1)
    out = cv2.remap(out, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    noise = np.random.randint(0, 40, (h // 2, w // 2), dtype=np.uint8)
    out = cv2.add(out, cv2.merge([cv2.resize(noise, (w, h))] * 3))
    out[::2] = (out[::2] * 0.82).astype(np.uint8)
    out = cv2.addWeighted(out, 0.9, np.full_like(out, (30, 10, 20)), 0.1, 0)
    q = c.get("quad")
    x0, y0 = (int(q[3][0]) + int(20 * S), int(q[3][1]) - int(24 * S)) if q is not None else (40, h - 40)
    cv2.putText(out, time.strftime("PLAY  %d.%m.%Y  %H:%M"), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * S,
                (255, 255, 255), max(1, int(2 * S)), cv2.LINE_AA)
    return out


def popart(img, c):
    """Pop art façon Warhol : le contenu du cadre en 4 vignettes aux couleurs saturées."""
    h, w = img.shape[:2]
    q = c.get("quad")
    if q is not None:
        x0, y0 = np.clip(q.min(axis=0).astype(int), 0, [w - 2, h - 2])
        x1, y1 = np.clip(q.max(axis=0).astype(int), [x0 + 2, y0 + 2], [w, h])
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    bw, bh = x1 - x0, y1 - y0
    tw, th = max(1, bw // 2), max(1, bh // 2)
    crop = cv2.resize(img[y0:y1, x0:x1], (tw, th), interpolation=cv2.INTER_AREA)
    g = cv2.equalizeHist(cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (0, 0), 1.2))
    lv = np.clip(g // 64, 0, 3)
    palettes = [[(80, 20, 140), (40, 120, 255), (60, 220, 255), (220, 250, 255)],
                [(120, 40, 20), (230, 60, 200), (90, 230, 120), (250, 250, 200)],
                [(20, 20, 20), (200, 40, 60), (40, 200, 255), (255, 255, 255)],
                [(90, 0, 60), (0, 140, 255), (255, 200, 0), (200, 255, 240)]]
    k = int(c["t"] * 0.7) % 4
    palettes = palettes[k:] + palettes[:k]
    out = img.copy()
    for i, pal in enumerate(palettes):
        ty, tx = y0 + (i // 2) * th, x0 + (i % 2) * tw
        tile = np.array(pal, np.uint8)[lv]
        out[ty:ty + th, tx:tx + tw] = tile[:max(0, min(th, h - ty)), :max(0, min(tw, w - tx))]
    return out


def ripple(img, c):
    """Ondulation : des vagues partent du centre du cadre comme dans l'eau."""
    h, w = img.shape[:2]
    cx, cy = c["center"] if c.get("center") is not None else (w / 2, h / 2)
    xx, yy = _grid(w // 2, h // 2)
    dx, dy = xx * 2 - cx, yy * 2 - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    S = _scale(img)
    amp = (6 + c["cell"] * 0.5) * S * (1 + 2 * c.get("audio", 0.0))
    disp = np.sin(r / (22 * S) - c["t"] * 6) * amp * np.exp(-r / (700 * S))
    mapx = cv2.resize((cx + dx + dx / r * disp).astype(np.float32), (w, h))
    mapy = cv2.resize((cy + dy + dy / r * disp).astype(np.float32), (w, h))
    out = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    shade = cv2.resize(((np.cos(r / (22 * S) - c["t"] * 6) * 18 * np.exp(-r / (700 * S))) + 128).astype(np.uint8),
                       (w, h))
    return cv2.add(cv2.subtract(out, 128 - np.minimum(shade, 128)[..., None].repeat(3, 2)),
                   np.maximum(shade, 128)[..., None].repeat(3, 2) - 128)


def _holo_lut():
    if "holo_lut" not in _cache:
        # noir → bleu nuit → bleu électrique → cyan → blanc bleuté (BGR)
        stops = [(0, (8, 2, 0)), (70, (120, 40, 5)), (140, (255, 140, 30)), (200, (255, 230, 120)),
                 (255, (255, 255, 235))]
        lut = np.zeros((256, 1, 3), np.uint8)
        for (a, ca), (b, cb) in zip(stops, stops[1:]):
            for v in range(a, b + 1):
                k = (v - a) / max(1, b - a)
                lut[v, 0] = [int(ca[i] + (cb[i] - ca[i]) * k) for i in range(3)]
        _cache["holo_lut"] = lut
    return _cache["holo_lut"]


def hologram(img, c):
    """Hologramme façon projection SF : ta silhouette seule, translucide et bleue, contours lumineux,
    aberration chromatique, lignes de balayage, onde d'interférence, décrochages, projecteur au sol."""
    h, w = img.shape[:2]
    t, S = c["t"], _scale(img)
    m = c.get("mask")
    if m is None:
        m = np.ones((h, w), np.float32)
    g = _clahe(2.0, (8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    holo = cv2.LUT(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), _holo_lut())
    # contours lumineux
    e = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.2), 40, 110)
    e = cv2.dilate(e, None)
    glow_e = cv2.addWeighted(_soft_blur(e, 3 * S), 2.0, e, 0.8, 0)
    holo = cv2.add(holo, cv2.merge([glow_e, cv2.convertScaleAbs(glow_e, alpha=0.9), cv2.convertScaleAbs(glow_e, alpha=0.4)]))
    # aberration chromatique (bleu et rouge décalés)
    k = max(1, int(3 * S))
    b_, g_, r_ = cv2.split(holo)
    holo = cv2.merge([np.roll(b_, k, axis=1), g_, np.roll(r_, -k, axis=1)])
    # lignes de balayage + onde d'interférence qui descend
    rows = np.arange(h, dtype=np.float32)
    period = max(3.0, 3.5 * S)
    scan = 0.62 + 0.38 * (np.sin((rows + t * 60 * S) / period * np.pi) > -0.2)
    band_y = ((t * 0.35) % 1.4 - 0.2) * h
    wave = 1 + 0.9 * np.exp(-((rows - band_y) / (40 * S)) ** 2)
    row_gain = (scan * wave).astype(np.float32)
    # décrochages : quelques bandes horizontales décalées de temps en temps
    rng = np.random.default_rng(int(t * 8))
    if rng.random() < 0.35:
        for _ in range(int(rng.integers(1, 4))):
            y0 = int(rng.integers(0, h))
            hh = int(rng.integers(6, 40) * S)
            dx = int(rng.integers(-30, 30) * S)
            holo[y0:y0 + hh] = np.roll(holo[y0:y0 + hh], dx, axis=1)
            m = m.copy()
            m[y0:y0 + hh] = np.roll(m[y0:y0 + hh], dx, axis=1)
    flick = 0.86 + 0.14 * math.sin(t * 31) * math.sin(t * 7.3) + rng.normal(0, 0.02)
    # opacité de l'hologramme (silhouette x lignes de balayage x scintillement), en 8 bits pour la vitesse
    a8 = np.clip(m * (0.85 * flick * 255) * row_gain[:, None], 0, 255).astype(np.uint8)
    holo_a = cv2.multiply(holo, cv2.cvtColor(a8, cv2.COLOR_GRAY2BGR), scale=1 / 255)
    # fond : la scène assombrie et bleutée, visible "à travers" l'hologramme
    out = cv2.add(cv2.convertScaleAbs(img, alpha=0.22), (40, 12, 0, 0))
    out = cv2.add(out, holo_a)
    # projecteur : disque lumineux sous la silhouette + faisceau
    ys, xs = np.nonzero(m[::8, ::8] > 0.5)
    if len(xs) > 20:
        cx = float(np.median(xs)) * 8
        top, bottom = float(ys.min()) * 8, min(h - 1.0, float(ys.max()) * 8 + 20 * S)
        width = (np.percentile(xs, 95) - np.percentile(xs, 5)) * 8 * 0.6 + 40 * S
        beam = np.zeros((h // 4, w // 4), np.float32)
        pts = np.array([[cx - width * 0.25, bottom], [cx + width * 0.25, bottom], [cx + width * 0.75, top],
                        [cx - width * 0.75, top]], np.float32) / 4
        cv2.fillConvexPoly(beam, pts.astype(np.int32), 0.16, cv2.LINE_AA)
        beam = cv2.GaussianBlur(beam, (0, 0), 6)
        disc = np.zeros_like(beam)
        cv2.ellipse(disc, (int(cx / 4), int(bottom / 4)), (int(width / 4 * 0.45), int(width / 4 * 0.09)), 0, 0, 360,
                    1.0, -1, cv2.LINE_AA)
        disc = cv2.GaussianBlur(disc, (0, 0), 3) * (0.8 + 0.2 * math.sin(t * 9))
        light = np.clip((beam + disc) * 255, 0, 255).astype(np.uint8)
        light = cv2.resize(light, (w, h), interpolation=cv2.INTER_LINEAR)
        out = cv2.add(out, cv2.merge([light, cv2.convertScaleAbs(light, alpha=0.82),
                                      cv2.convertScaleAbs(light, alpha=0.35)]))
        # particules qui montent dans le faisceau
        st = c["state"].setdefault("holo", {})
        if "p" not in st:
            st["p"] = np.random.default_rng(1).uniform(0, 1, (90, 3)).astype(np.float32)
        p = st["p"]
        p[:, 1] = (p[:, 1] + min(c.get("dt", 0.033), 0.1) * (0.2 + p[:, 2] * 0.3)) % 1.0
        for px, py, pz in p.tolist():
            yy = bottom - py * (bottom - top)
            xx = cx + (px - 0.5) * width * (0.5 + py)
            a_ = (1 - py) * 0.8
            cv2.circle(out, (int(xx), int(yy)), max(1, int(2 * S * (0.5 + pz))), (255 * a_, 230 * a_, 140 * a_), -1,
                       cv2.LINE_AA)
    noise = np.random.randint(0, 14, (h // 2, w // 2), dtype=np.uint8)
    out = cv2.add(out, cv2.cvtColor(cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR))
    return cv2.subtract(out, (7, 7, 7, 0))


def slitscan(img, c):
    """Slit-scan : chaque bande horizontale vient d'un instant différent → corps qui se tordent."""
    hist = c.get("history")
    if hist is None or not hist.buf:
        return img
    h, w = img.shape[:2]
    bands = 36
    out = np.empty_like(img)
    bh = int(np.ceil(h / bands))
    for i in range(bands):
        f, _ = hist.at(i / bands * 1.6)
        f = img if f is None else f
        out[i * bh:(i + 1) * bh] = f[i * bh:(i + 1) * bh]
    return out


def cartoon(img, c):
    """Cartoon : aplats de couleurs + contours encrés."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    for _ in range(2):
        small = cv2.bilateralFilter(small, 7, 60, 7)
    q = (small // 40) * 40 + 20
    col = cv2.resize(q, (w, h), interpolation=cv2.INTER_LINEAR)
    g = cv2.medianBlur(cv2.cvtColor(cv2.resize(img, (w // 2, h // 2)), cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 4)
    edges = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.bitwise_and(col, cv2.merge([edges] * 3))


# =========================================================================== effets photo réalistes
def _curve(points):
    """Courbe tonale 0..255 à partir de points de contrôle (comme dans Lightroom)."""
    xs, ys = zip(*points)
    return np.clip(np.interp(np.arange(256), xs, ys), 0, 255).astype(np.uint8)


def _lut3(b, g, r):
    return np.dstack([b, g, r]).reshape(256, 1, 3)


def _grain(img, amount):
    h, w = img.shape[:2]
    key = ("grain2", w, h)
    if key not in _cache:
        n = np.random.default_rng(5).normal(0, 1, (h + 64, w + 64)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 0.7 * _scale(img))
        _cache[key] = np.clip(n * 40 + 128, 0, 255).astype(np.uint8)
    n = _cache[key]
    oy, ox = np.random.randint(0, 64, 2)
    g = cv2.cvtColor(n[oy:oy + h, ox:ox + w], cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 1.0, g, amount, -128 * amount)


def _vignette(img, strength=0.35, key_name="vig2"):
    h, w = img.shape[:2]
    key = (key_name, w, h, strength)
    if key not in _cache:
        yy, xx = np.mgrid[0:h // 4, 0:w // 4].astype(np.float32)
        d = np.sqrt(((xx - w / 8) / (w / 8)) ** 2 + ((yy - h / 8) / (h / 8)) ** 2) / 1.414
        v = np.clip(1 - strength * d ** 2.2, 0, 1)
        _cache[key] = cv2.cvtColor((cv2.resize(v, (w, h)) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return cv2.multiply(img, _cache[key], scale=1 / 255)


def _soft_mask(m, w, h):
    ms = cv2.GaussianBlur(cv2.resize(m, (w // 4, h // 4), interpolation=cv2.INTER_AREA), (0, 0), 1.2)
    return np.clip(cv2.resize(ms, (w, h), interpolation=cv2.INTER_LINEAR), 0, 1)


def cinema(img, c):
    """Cinéma : étalonnage bleu-orangé (ombres froides, peaux et hautes lumières chaudes), bandes 2.39:1, grain."""
    if "cine_lut" not in _cache:
        _cache["cine_lut"] = _lut3(_curve([(0, 18), (70, 80), (160, 160), (255, 232)]),
                                   _curve([(0, 8), (64, 58), (190, 196), (255, 245)]),
                                   _curve([(0, 0), (64, 52), (180, 200), (255, 255)]))
    out = cv2.LUT(img, _cache["cine_lut"])
    out = _vignette(_grain(out, 0.12), 0.3)
    q = c.get("quad")
    if q is not None:  # bandes noires calées sur le cadre
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        bar = max(0.0, ((y1 - y0) - (x1 - x0) / 2.39) / 2)
        if bar > 1:
            out[int(max(0, y0)):int(y0 + bar)] = 0
            out[int(y1 - bar):int(min(img.shape[0], y1)) + 1] = 0
    return out


def portrait(img, c):
    """Portrait : arrière-plan flouté avec un vrai bokeh d'objectif (disques lumineux), toi bien net."""
    m = c.get("mask")
    h, w = img.shape[:2]
    sw, sh = w // 4, h // 4
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    r = max(3, int(round(sh / 45)))
    key = ("disc", r)
    if key not in _cache:
        k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        cv2.circle(k, (r, r), r, 1.0, -1, cv2.LINE_AA)
        _cache[key] = k / k.sum()
    lin = small ** 2.4  # en lumière "linéaire" : les points lumineux deviennent de beaux disques
    bl = cv2.filter2D(lin, -1, _cache[key]) ** (1 / 2.4)
    bg = cv2.resize(np.clip(bl * 255, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    if m is None:
        return bg
    mf = _soft_mask(m, w, h)
    return cv2.blendLinear(img, bg, mf, 1.0 - mf)


def depth_blur(img, m, strength):
    """Profondeur automatique : flou d'arrière-plan façon objectif (bokeh), dosé par `strength` (0..1)."""
    h, w = img.shape[:2]
    sw, sh = w // 4, h // 4
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    r = max(2, int(round(sh / 45 * (0.45 + 0.55 * strength))))
    key = ("disc", r)
    if key not in _cache:
        k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        cv2.circle(k, (r, r), r, 1.0, -1, cv2.LINE_AA)
        _cache[key] = k / k.sum()
    bl = cv2.filter2D(small ** 2.4, -1, _cache[key]) ** (1 / 2.4)
    bg = cv2.resize(np.clip(bl * 255, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    mf = _soft_mask(m, w, h)
    a = mf + (1.0 - mf) * (1.0 - min(1.0, strength))  # part de l'image nette
    return cv2.blendLinear(img, bg, a, 1.0 - a)


def studio_light(img, c):
    """Studio : fond de studio sombre et dégradé, toi éclairé en douceur (façon portrait de studio)."""
    m = c.get("mask")
    h, w = img.shape[:2]
    key = ("backdrop", w, h)
    if key not in _cache:
        yy, xx = np.mgrid[0:h // 4, 0:w // 4].astype(np.float32)
        d = np.sqrt(((xx - w / 8) / (w / 8)) ** 2 + ((yy - h / 10) / (h / 6)) ** 2)
        v = np.clip(62 - 48 * d, 10, 62)
        bd = cv2.resize(np.dstack([v * 1.05, v, v * 0.95]), (w, h), interpolation=cv2.INTER_CUBIC)
        _cache[key] = np.clip(bd + np.random.default_rng(2).uniform(-1.5, 1.5, (h, w, 1)), 0, 255).astype(np.uint8)
    lit = cv2.LUT(img, _cache.setdefault("studio_lut", _lut3(*[_curve([(0, 4), (60, 62), (180, 200), (255, 252)])] * 3)))
    if m is None:
        return lit
    mf = _soft_mask(m, w, h)
    return cv2.blendLinear(lit, _cache[key], mf, 1.0 - mf)


def film_bw(img, c):
    """Argentique : noir et blanc de pellicule (filtre rouge doux, contraste en S, grain, vignettage)."""
    b, g, r = cv2.split(img)
    lum = cv2.addWeighted(cv2.addWeighted(r, 0.45, g, 0.45, 0), 1.0, b, 0.1, 0)
    lum = cv2.LUT(lum, _cache.setdefault("bw_lut", _curve([(0, 6), (50, 34), (128, 132), (205, 222), (255, 250)])))
    out = cv2.cvtColor(lum, cv2.COLOR_GRAY2BGR)
    return _vignette(_grain(out, 0.22), 0.45, "vigbw")


def vintage(img, c):
    """Vintage : film couleur des années 70 (noirs délavés, hautes lumières chaudes, un peu désaturé, grain)."""
    if "vint_lut" not in _cache:
        _cache["vint_lut"] = _lut3(_curve([(0, 40), (128, 118), (255, 205)]),
                                   _curve([(0, 30), (128, 132), (255, 235)]),
                                   _curve([(0, 26), (128, 142), (255, 250)]))
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(img, 0.78, gray, 0.22, 0)
    out = cv2.LUT(out, _cache["vint_lut"])
    return _vignette(_grain(out, 0.16), 0.4, "vigvint")


def golden_hour(img, c):
    """Golden hour : lumière dorée de fin de journée, hautes lumières qui rayonnent (halo doux)."""
    if "gold_lut" not in _cache:
        _cache["gold_lut"] = _lut3(_curve([(0, 0), (128, 116), (255, 232)]),
                                   _curve([(0, 3), (128, 128), (255, 248)]),
                                   _curve([(0, 8), (128, 141), (255, 255)]))
    out = cv2.LUT(img, _cache["gold_lut"])
    bright = cv2.subtract(out, (150, 150, 150, 0))
    glow = _soft_blur(bright, 22 * _scale(img))
    out = cv2.addWeighted(out, 1.0, glow, 0.8, 0)
    return _vignette(out, 0.28, "viggold")


# =========================================================================== 4K
FX_MAX_H = 1440  # au-delà, les effets "doux" sont calculés en demi-résolution puis agrandis
NATIVE = {"pixel", "ascii", "matrix", "halftone", "tunnel infini", "invert", "miroir"}  # finesse visible en 4K


class _HistScaled:
    """Mémoire d'images vue à une autre taille (pour les effets calculés en demi-résolution)."""

    def __init__(self, hist, size):
        self.hist, self.size = hist, size

    @property
    def buf(self):
        return self.hist.buf

    def at(self, delay):
        f, m = self.hist.at(delay)
        if f is not None and (f.shape[1], f.shape[0]) != self.size:
            f = cv2.resize(f, self.size, interpolation=cv2.INTER_AREA)
        return f, m


# effets très "doux" : calculés encore plus petit (invisible à l'œil, mais 2 à 4x plus rapide)
FX_CAP = {"portrait": 1080, "studio": 1080, "golden hour": 1080, "hologramme": 540, "vhs": 540, "vision nuit": 540, "ondulation": 720, "croquis": 720, "cartoon": 720,
          "thermique": 540, "neon": 720, "kaléidoscope": 720, "slit-scan": 720, "clones": 720, "invisibilité": 720}


def _halfres(fn, cap=None):
    cap = cap or FX_MAX_H

    def run(img, c):
        h, w = img.shape[:2]
        if h > cap and h <= FX_MAX_H or h > FX_MAX_H:
            f = 2
            while h / f > cap:
                f *= 2
            sw, sh = w // f, h // f
            small = img
            for _ in range(int(math.log2(f))):
                small = cv2.pyrDown(small)
            if small.shape[:2] != (sh, sw):
                small = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_AREA)
            c2 = dict(c)
            for k in ("quad", "center"):
                if c.get(k) is not None:
                    c2[k] = np.asarray(c[k], np.float32) / f
            if c.get("mask") is not None:
                c2["mask"] = cv2.resize(c["mask"], (sw, sh), interpolation=cv2.INTER_AREA)
            for k in ("prev", "bg"):
                if c.get(k) is not None:
                    c2[k] = cv2.resize(c[k], (sw, sh), interpolation=cv2.INTER_AREA)
            if c.get("history") is not None:
                c2["history"] = _HistScaled(c["history"], (sw, sh))
            return cv2.resize(fn(small, c2), (w, h), interpolation=cv2.INTER_LINEAR)
        if h <= FX_MAX_H:
            hist = c.get("history")
            if hist is not None and hist.buf and hist.buf[-1][1].shape[:2] != img.shape[:2]:
                c = dict(c, history=_HistScaled(hist, (w, h)))
            return fn(img, c)
        sw, sh = w // 2, h // 2
        small = cv2.pyrDown(img) if (w % 2 == 0 and h % 2 == 0) else _shrink(img, (sw, sh))
        c2 = dict(c)
        for k in ("quad", "center"):
            if c.get(k) is not None:
                c2[k] = np.asarray(c[k], np.float32) * 0.5
        if c.get("mask") is not None:
            c2["mask"] = cv2.resize(c["mask"], (sw, sh), interpolation=cv2.INTER_AREA)
        for k in ("prev", "bg"):
            if c.get(k) is not None:
                c2[k] = cv2.resize(c[k], (sw, sh), interpolation=cv2.INTER_AREA)
        if c.get("history") is not None:
            c2["history"] = _HistScaled(c["history"], (sw, sh))
        return cv2.resize(fn(small, c2), (w, h), interpolation=cv2.INTER_LINEAR)
    run.__name__, run.__doc__ = fn.__name__, fn.__doc__
    return run


# nom, fonction, besoins ("seg" = silhouette, "hist" = mémoire d'images, "bg" = fond)
EFFECTS = [
    ("pixel", pixel, ()),
    ("flou", frosted, ()),
    ("ascii", ascii_art, ()),
    ("matrix", matrix, ()),
    ("tunnel infini", tunnel, ()),
    ("portail temporel", portal, ("hist",)),
    ("invisibilité", invisible, ("seg", "bg")),
    ("clones", clones, ("seg", "hist")),
    ("aura", aura, ("seg",)),
    ("espace", space, ("seg",)),
    ("thermique", thermal, ("seg",)),
    ("neon", neon, ()),
    ("glitch", glitch, ()),
    ("kaléidoscope", kaleido, ()),
    ("bulle", bubble, ()),
    ("rgb temporel", rgb_time, ("hist",)),
    ("croquis", sketch, ()),
    ("halftone", halftone, ()),
    ("vision nuit", night, ()),
    ("miroir", mirror, ()),
    ("vhs", vhs, ()),
    ("pop art", popart, ()),
    ("ondulation", ripple, ()),
    ("hologramme", hologram, ("seg",)),
    ("slit-scan", slitscan, ("hist",)),
    ("cartoon", cartoon, ()),
    ("invert", invert, ()),
]

from armor import armure  # noqa: E402  (armure high-tech : casque, réacteur, propulseurs)

EFFECTS.insert(1, ("armure", armure, ("full", "face", "pose")))
NATIVE.add("armure")
for _i, _fx in enumerate([("portrait", portrait, ("seg",)), ("cinéma", cinema, ()), ("studio", studio_light, ("seg",)),
                          ("argentique", film_bw, ()), ("vintage", vintage, ()), ("golden hour", golden_hour, ())]):
    EFFECTS.insert(2 + _i, _fx)
EFFECTS = [fx for fx in EFFECTS if fx[0] != "espace"]  # retiré : pas assez réaliste

# Effets personnels optionnels : un fichier local_effects.py placé à côté (ignoré par git, jamais publié)
# peut fournir LOCAL_EFFECTS = [(nom, fonction, besoins), ...] et LOCAL_NATIVE = {noms en résolution native}.
try:
    import local_effects  # noqa: E402
    for _i, _fx in enumerate(getattr(local_effects, "LOCAL_EFFECTS", [])):
        EFFECTS.insert(2 + _i, _fx)
    NATIVE |= set(getattr(local_effects, "LOCAL_NATIVE", ()))
except ImportError:
    pass

EFFECTS = [(n, fn if n in NATIVE else _halfres(fn, FX_CAP.get(n)), needs) for n, fn, needs in EFFECTS]
