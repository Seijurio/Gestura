"""Lumière d'appoint automatique (« ring light » d'écran) + correction d'exposition en faible lumière.

- Mesure la luminosité de l'image caméra.
- Si c'est sombre : l'image affichée rétrécit et s'entoure d'un grand cadre blanc chaud, très lumineux,
  qui éclaire ton visage (l'écran sert de lampe). En plein écran (touche E), c'est encore plus puissant.
- En parallèle, l'exposition est relevée (ombres éclaircies) sur l'image de travail.
- Le cadre lumineux n'est QUE à l'affichage : il n'apparaît jamais sur les photos ni les vidéos.

Modes (touche I) : auto → toujours allumée → éteinte.
"""
import math

import cv2
import numpy as np

MODES = ("auto", "on", "off")
LABELS = {"auto": "Lumière d'appoint : auto", "on": "Lumière d'appoint : allumée", "off": "Lumière d'appoint : éteinte"}
WARM_WHITE = (238, 246, 255)  # BGR : blanc légèrement chaud, flatteur pour la peau


class RingLight:
    def __init__(self, mode="auto"):
        self.mode = mode if mode in MODES else "auto"
        self.lum = 128.0   # luminosité moyenne mesurée (0..255)
        self.level = 0.0   # puissance actuelle de la lumière d'écran (0..1)
        self._luts = {}

    def cycle(self):
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        return LABELS[self.mode]

    # ------------------------------------------------------------------ mesure
    def measure(self, frame):
        """À appeler à chaque nouvelle image caméra (sur l'image brute, avant correction)."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame[h // 8:h * 7 // 8, w // 6:w * 5 // 6], (96, 54), interpolation=cv2.INTER_AREA)
        y = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
        self.lum = 0.85 * self.lum + 0.15 * y

    def target(self):
        if self.mode == "off":
            return 0.0
        if self.mode == "on":
            return 1.0
        # auto : rien au-dessus de 100, puissance max en dessous de 45
        return float(np.clip((100 - self.lum) / 55, 0, 1))

    def update(self, dt):
        t = self.target()
        speed = 2.5 if t > self.level else 1.2  # s'allume vite, s'éteint doucement
        self.level += (t - self.level) * (1 - math.exp(-speed * dt))

    # ------------------------------------------------------------------ correction d'exposition
    def expose(self, frame):
        """Relève l'exposition quand l'image est sombre (ombres surtout), sans brûler les hautes lumières."""
        if self.mode == "off" or self.lum >= 95:
            return frame
        gain = float(np.clip(105 / max(self.lum, 8), 1.0, 2.6))
        key = round(gain, 1)
        if key not in self._luts:
            x = np.arange(256, dtype=np.float32) / 255
            # courbe "gain + épaule douce" : les ombres montent beaucoup, les blancs restent blancs
            y = 1 - np.exp(-x * key * 1.6)
            y = y / (1 - math.exp(-key * 1.6))
            self._luts[key] = np.clip(y * 255, 0, 255).astype(np.uint8)
        return cv2.LUT(frame, self._luts[key])

    # ------------------------------------------------------------------ anneau lumineux (affichage)
    def _ring_assets(self, w, h, b):
        """Anneau (dégradé façon LED), masque à bords doux et ombre intérieure : calculés une fois par taille."""
        key = ("ring", w, h, b)
        if key in self._luts:
            return self._luts[key]
        ih = h - 2 * b
        iw = int(ih * w / h)
        x0, y0 = (w - iw) // 2, b
        r = max(6, int(ih * 0.04))
        # masque de l'image (coins arrondis, bord adouci)
        m = np.zeros((ih, iw), np.uint8)
        cv2.rectangle(m, (r, 0), (iw - r, ih), 255, -1)
        cv2.rectangle(m, (0, r), (iw, ih - r), 255, -1)
        for cx, cy in ((r, r), (iw - r - 1, r), (r, ih - r - 1), (iw - r - 1, ih - r - 1)):
            cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        mf = (m.astype(np.float32) / 255)[..., None]
        # ombre intérieure : l'image paraît "enfoncée" dans l'anneau (relief)
        dist = cv2.distanceTransform((m > 127).astype(np.uint8), cv2.DIST_L2, 3)
        inner = (1 - 0.38 * np.exp(-dist / max(4.0, b * 0.5)))[..., None].astype(np.float32)
        # anneau : plus lumineux au milieu de sa largeur (comme une bande de LED sous diffuseur)
        full = np.zeros((h, w), np.uint8)
        full[y0:y0 + ih, x0:x0 + iw] = (m > 127).astype(np.uint8)
        d_out = cv2.distanceTransform(1 - full, cv2.DIST_L2, 3)  # distance au bord de l'image
        prof = np.clip(1 - np.abs(d_out - b * 0.5) / (b * 0.75), 0, 1) ** 0.7
        prof = np.where(full > 0, 0.0, 0.55 + 0.45 * prof)
        ring = (prof[..., None] * np.array(WARM_WHITE, np.float32)).astype(np.uint8)
        ring_dim = cv2.convertScaleAbs(ring, alpha=0.18)  # fond sombre autour quand la puissance est faible
        w_img = np.clip(inner * mf * 255, 0, 255).astype(np.uint8).repeat(3, axis=2)  # poids 8 bits (rapide)
        w_bg = np.clip((1 - mf) * 255, 0, 255).astype(np.uint8).repeat(3, axis=2)
        self._luts[key] = (x0, y0, iw, ih, w_img, w_bg, ring, ring_dim)
        return self._luts[key]

    def apply(self, img):
        """Entoure l'image d'un anneau lumineux fin, dont la luminosité suit la puissance demandée."""
        if self.level < 0.02:
            return img
        h, w = img.shape[:2]
        b = int(h * (0.022 + 0.028 * self.level)) // 4 * 4  # 2 à 5 % de la hauteur, par pas de 4 px
        x0, y0, iw, ih, w_img, w_bg, ring, ring_dim = self._ring_assets(w, h, b)
        k = 0.35 + 0.65 * self.level
        canvas = cv2.addWeighted(ring, k, ring_dim, 1 - k, 0)
        small = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_LINEAR)
        roi = canvas[y0:y0 + ih, x0:x0 + iw]
        roi[:] = cv2.add(cv2.multiply(small, w_img, scale=1 / 255), cv2.multiply(roi, w_bg, scale=1 / 255))
        return canvas
