"""Dessin néon dans l'air.

  ☝️ index tendu (ou 🤏 pince)  → dessine
  ✋ main ouverte               → gomme (autour de la paume)
  barre d'outils en haut        → pointe une couleur / taille / outil avec l'index 0,4 s pour le choisir

Les traits sont des courbes vectorielles lissées (Chaikin), dessinées en néon avec un halo.
Les traits terminés sont « cuits » dans un calque en cache : seul le trait en cours est redessiné.
"""
import math
import time

import cv2
import numpy as np

import ui
from tracking import Smoother

PALETTE = [("arc-en-ciel", None), ("blanc", (255, 255, 255)), ("rouge", (70, 60, 255)),
           ("orange", (40, 150, 255)), ("jaune", (50, 235, 255)), ("vert", (90, 255, 110)),
           ("cyan", (255, 230, 70)), ("bleu", (255, 120, 50)), ("violet", (255, 80, 200))]
SIZES = [4, 8, 14]
TOOLS = ["annuler", "effacer"]
DWELL = 0.4


def _chaikin(pts, it=2):
    for _ in range(it):
        if len(pts) < 3:
            return pts
        q = pts[:-1] * 0.75 + pts[1:] * 0.25
        r = pts[:-1] * 0.25 + pts[1:] * 0.75
        mid = np.empty((len(q) * 2, 2), np.float32)
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


class Stroke:
    def __init__(self, color, size, hue0):
        self.pts = []
        self.color, self.size, self.hue0 = color, size, hue0


class AirCanvas:
    def __init__(self):
        self.strokes = []
        self.active = {}
        self.last_down = {}
        self.smooth = {}
        self.color_i, self.size_i = 0, 1
        self.cache = None
        self.dirty = True
        self.hover = {}
        self.cursors = []
        self.erasers = []

    # ------------------------------------------------------------------ barre d'outils
    def _items(self, w, h):
        """[(type, index, (x, y), rayon)]"""
        r = ui.u(17)
        gap = ui.u(46)
        n = len(PALETTE) + len(SIZES) + len(TOOLS)
        x = w // 2 - (n - 1) * gap // 2 - ui.u(30)
        y = ui.u(98)
        out = []
        for i in range(len(PALETTE)):
            out.append(("color", i, (x, y), r))
            x += gap
        x += ui.u(20)
        for i in range(len(SIZES)):
            out.append(("size", i, (x, y), r))
            x += gap
        x += ui.u(20)
        for i in range(len(TOOLS)):
            out.append(("tool", i, (x + ui.u(14), y), r + ui.u(14)))
            x += gap + ui.u(40)
        return out

    def _toolbar_bottom(self):
        return ui.u(98) + ui.u(40)

    def _apply(self, item):
        kind, i = item[0], item[1]
        if kind == "color":
            self.color_i = i
            return f"Couleur : {PALETTE[i][0]}"
        if kind == "size":
            self.size_i = i
            return f"Taille : {['fin', 'moyen', 'épais'][i]}"
        if TOOLS[i] == "annuler":
            self.undo()
            return "Annulé"
        self.clear()
        return "Dessin effacé"

    def undo(self):
        if self.strokes:
            self.strokes.pop()
            self.dirty = True

    def clear(self):
        self.strokes, self.active = [], {}
        self.dirty = True

    # ------------------------------------------------------------------ mise à jour
    def update(self, poses, w, h):
        """poses : {clé: (pose, main)}. Renvoie un message (toast) ou None."""
        now = time.monotonic()
        msg = None
        self.cursors, self.erasers = [], []
        items = self._items(w, h)
        seen = set()
        for key, (pose, hd) in poses.items():
            if hd is None:
                continue
            seen.add(key)
            if pose == "open":
                c, r = hd.pts[9], hd.palm * 0.85
                self.erasers.append((c, r))
                self._erase(c, r)
                self._end(key)
                continue
            down = pose in ("point", "pinch")
            raw = hd.pts[8] if pose != "pinch" else hd.pinch_point
            sm = self.smooth.setdefault(key, Smoother(0.3, 0.9, 25 * ui.S))
            p = sm.update(raw)
            self.cursors.append((p, down))
            if not down:
                if now - self.last_down.get(key, 0) > 0.15:
                    self._end(key)
                self.hover.pop(key, None)
                continue
            self.last_down[key] = now
            if p[1] < self._toolbar_bottom():  # dans la barre d'outils : choix au lieu de dessiner
                self._end(key)
                hit = None
                for it in items:
                    if math.hypot(p[0] - it[2][0], p[1] - it[2][1]) < it[3] + ui.u(8):
                        hit = it
                if hit is None:
                    self.hover.pop(key, None)
                else:
                    cur = self.hover.get(key)
                    if cur is None or cur[0][:2] != hit[:2]:
                        self.hover[key] = (hit, now, False)
                    elif not cur[2] and now - cur[1] >= DWELL:
                        self.hover[key] = (hit, cur[1], True)
                        msg = self._apply(hit)
                continue
            self.hover.pop(key, None)
            st = self.active.get(key)
            if st is None:
                col = PALETTE[self.color_i][1]
                st = self.active[key] = Stroke(col, SIZES[self.size_i], (now * 0.1) % 1.0)
            if st.pts:
                last = st.pts[-1]
                dist = math.hypot(p[0] - last[0], p[1] - last[1])
                if dist > 0.25 * w:  # saut (perte de suivi) : on commence un nouveau trait
                    self._end(key)
                    continue
                if dist < 1.5 * ui.S:
                    continue
            st.pts.append((float(p[0]), float(p[1])))
        for key in list(self.active):
            if key not in seen and now - self.last_down.get(key, 0) > 0.15:
                self._end(key)
        for key in list(self.smooth):
            if key not in seen:
                del self.smooth[key]
        return msg

    def _end(self, key):
        st = self.active.pop(key, None)
        if st is not None and len(st.pts) >= 2:
            self.strokes.append(st)
            self.dirty = True

    def _erase(self, c, r):
        changed = False
        out = []
        for st in self.strokes:
            pts = np.array(st.pts, np.float32)
            keep = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) > r
            if keep.all():
                out.append(st)
                continue
            changed = True
            run = []
            for pt, k in zip(st.pts, keep):
                if k:
                    run.append(pt)
                elif run:
                    if len(run) >= 2:
                        ns = Stroke(st.color, st.size, st.hue0)
                        ns.pts = run
                        out.append(ns)
                    run = []
            if len(run) >= 2:
                ns = Stroke(st.color, st.size, st.hue0)
                ns.pts = run
                out.append(ns)
        if changed:
            self.strokes = out
            self.dirty = True

    # ------------------------------------------------------------------ rendu
    def _render_stroke(self, img, st, k):
        pts = _chaikin(np.array(st.pts, np.float32)) * k
        th = max(1, int(st.size * ui.S * k * 1.6))
        core = max(1, th // 3)
        if st.color is not None:
            q = pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [q], False, st.color, th, cv2.LINE_AA)
            cv2.polylines(img, [q], False, (255, 255, 255), core, cv2.LINE_AA)
            return
        seg = np.hypot(*(pts[1:] - pts[:-1]).T)
        cum = np.concatenate([[0], np.cumsum(seg)]) / (400 * ui.S * k)
        for i in range(len(pts) - 1):
            a, b = tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int))
            cv2.line(img, a, b, ui.hsv_color(st.hue0 + cum[i], 0.85, 1.0), th, cv2.LINE_AA)
        q = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [q], False, (255, 255, 255), core, cv2.LINE_AA)

    def _layer(self, strokes, w, h):
        k = 0.5
        layer = np.zeros((h // 2, w // 2, 3), np.uint8)
        for st in strokes:
            self._render_stroke(layer, st, k)
        glow = ui.soft_blur(layer, 8 * ui.S * k)
        return cv2.addWeighted(layer, 1.0, glow, 1.8, 0)

    def render(self, frame, show_ui):
        h, w = frame.shape[:2]
        if self.dirty:
            self.cache = self._layer(self.strokes, w, h) if self.strokes else None
            self.dirty = False
        layer = self.cache
        act = [s for s in self.active.values() if len(s.pts) >= 2]
        if act:
            al = self._layer(act, w, h)
            layer = al if layer is None else cv2.add(layer, al)
        if layer is not None:
            cv2.add(frame, cv2.resize(layer, (w, h), interpolation=cv2.INTER_LINEAR), frame)
        if show_ui:
            self._draw_ui(frame)

    def _draw_ui(self, frame):
        h, w = frame.shape[:2]
        items = self._items(w, h)
        x0 = min(it[2][0] - it[3] for it in items) - ui.u(18)
        x1 = max(it[2][0] + it[3] for it in items) + ui.u(18)
        y = items[0][2][1]
        ui.panel(frame, x0, y - ui.u(30), x1, y + ui.u(30), 0.55, 26)
        hovered = {v[0][:2]: v for v in self.hover.values()}
        now = time.monotonic()
        for it in items:
            kind, i, (cx, cy), r = it
            sel = (kind == "color" and i == self.color_i) or (kind == "size" and i == self.size_i)
            if kind == "color":
                col = PALETTE[i][1]
                if col is None:
                    for k in range(12):
                        cv2.ellipse(frame, (cx, cy), (r, r), 0, k * 30, k * 30 + 31, ui.hsv_color(k / 12), -1,
                                    cv2.LINE_AA)
                else:
                    cv2.circle(frame, (cx, cy), r, col, -1, cv2.LINE_AA)
            elif kind == "size":
                cv2.circle(frame, (cx, cy), r, (60, 60, 60), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), max(2, int(SIZES[i] * ui.S * 0.8)), ui.WHITE, -1, cv2.LINE_AA)
            else:
                ui.Text_default().draw(frame, TOOLS[i], (cx, cy), 16, ui.WHITE, "mm", bold=True)
            if sel:
                cv2.circle(frame, (cx, cy), r + ui.u(5), ui.WHITE, max(1, ui.u(2)), cv2.LINE_AA)
            hv = hovered.get((kind, i))
            if hv is not None and not hv[2]:
                ui.arc(frame, (cx, cy), r + ui.u(9), (now - hv[1]) / DWELL, ui.GREEN, 3)
        col = PALETTE[self.color_i][1] or ui.hsv_color(now * 0.3)
        for p, down in self.cursors:
            c = (int(p[0]), int(p[1]))
            rr = max(ui.u(6), int(SIZES[self.size_i] * ui.S))
            if down:
                cv2.circle(frame, c, rr, col, -1, cv2.LINE_AA)
            cv2.circle(frame, c, rr + ui.u(5), ui.WHITE, max(1, ui.u(1.5)), cv2.LINE_AA)
        for c, r in self.erasers:
            cv2.circle(frame, (int(c[0]), int(c[1])), int(r), ui.WHITE, max(1, ui.u(2)), cv2.LINE_AA)
            ui.Text_default().draw(frame, "gomme", (c[0], c[1]), 16, ui.WHITE, "mm", bold=True, stroke=2)
