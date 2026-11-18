"""Effet « armure » : tu es dans le casque d'une armure high-tech (design original).

- interface de casque : démarrage, compas qui suit ta tête, radar, jauge de puissance, verrouillage
  de cible sur ton visage avec distance estimée, texte système ;
- réacteur lumineux sur la poitrine (suivi du corps), avec une vraie lueur qui éclaire la scène ;
- propulseurs : main ouverte face caméra → ça se charge (lueur + anneau), puis ça tire
  (éclair, onde de choc, traînée anamorphique, petit tremblement, son).
"""
import math
import time

import cv2
import numpy as np

import ui

ICE = (255, 215, 130)     # bleu glacier (BGR)
AMBER = (60, 185, 255)
CORE = (255, 250, 235)
_cache = {}


# =========================================================================== utilitaires
def _txt(img, s, pos, size, color=ICE, anchor="lt", alpha=0.9, bold=True):
    ui.Text_default().draw(img, s, pos, size, color, anchor, bold=bold, alpha=alpha)


def _emit(out, center, ext, draw_fn, sigma, gain):
    """Dessine un élément lumineux dans une petite zone, ajoute un halo flouté (lumière qui « bave »)."""
    h, w = out.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    ext = int(ext)
    x0, y0, x1, y1 = cx - ext, cy - ext, cx + ext, cy + ext
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if xa >= xb or ya >= yb:
        return
    patch = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    draw_fn(patch, (cx - x0, cy - y0))
    glow = ui.soft_blur(patch, sigma)
    lit = cv2.addWeighted(patch, 1.0, glow, gain, 0)
    out[ya:yb, xa:xb] = cv2.add(out[ya:yb, xa:xb], lit[ya - y0:yb - y0, xa - x0:xb - x0])


def _grade():
    if "lut" not in _cache:
        x = np.arange(256, dtype=np.float32)
        s = np.clip(255 * (0.5 + 0.5 * np.tanh((x / 255 - 0.5) * 2.4) / np.tanh(1.2)), 0, 255)
        b = np.clip(s * 1.04 + 6, 0, 255)
        g = np.clip(s * 0.98, 0, 255)
        r = np.clip(s * 0.9, 0, 255)
        _cache["lut"] = np.dstack([b, g, r]).astype(np.uint8).reshape(256, 1, 3)
    return _cache["lut"]


def _static(w, h):
    """Éléments fixes du casque (dessinés une fois) + vignettage bleuté du verre."""
    key = ("hud", w, h)
    if key in _cache:
        return _cache[key]
    S = h / 720
    th = max(1, int(1.4 * S))
    L = np.zeros((h, w, 3), np.uint8)
    # bord de la visière : arcs fins tout près des bords haut et bas, en pointillés
    for cy, a0, a1 in ((-h * 1.25, 35, 145), (h * 2.25, 215, 325)):
        for a_ in np.arange(a0, a1, 2.0):
            cv2.ellipse(L, (w // 2, int(cy)), (int(w * 0.78), int(h * 1.39)), 0, a_, a_ + 1.2, ICE, th, cv2.LINE_AA)
    # crochets de coin
    m, ln = int(0.035 * w), int(0.06 * w)
    for (x, y, dx, dy) in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
        cv2.line(L, (x, y), (x + dx * ln, y), ICE, th * 2, cv2.LINE_AA)
        cv2.line(L, (x, y), (x, y + dy * ln), ICE, th * 2, cv2.LINE_AA)
    # échelle verticale gauche
    x = int(0.06 * w)
    for i, y in enumerate(np.linspace(h * 0.3, h * 0.7, 41)):
        ll = int((14 if i % 5 == 0 else 7) * S)
        cv2.line(L, (x, int(y)), (x + ll, int(y)), ICE, th, cv2.LINE_AA)
    # radar (bas droite)
    rc, rr = (int(w * 0.9), int(h * 0.76)), int(h * 0.085)
    for k in (1.0, 0.66, 0.33):
        cv2.circle(L, rc, int(rr * k), ICE, th, cv2.LINE_AA)
    cv2.line(L, (rc[0] - rr, rc[1]), (rc[0] + rr, rc[1]), ICE, th, cv2.LINE_AA)
    cv2.line(L, (rc[0], rc[1] - rr), (rc[0], rc[1] + rr), ICE, th, cv2.LINE_AA)
    # réticule central discret
    c = (w // 2, h // 2)
    for a0 in (20, 110, 200, 290):
        cv2.ellipse(L, c, (int(28 * S), int(28 * S)), 0, a0, a0 + 50, ICE, th, cv2.LINE_AA)
    L = cv2.convertScaleAbs(L, alpha=0.75)
    # verre du casque : bords assombris et bleutés + reflet diagonal
    yy, xx = np.mgrid[0:h // 4, 0:w // 4].astype(np.float32)
    d = np.sqrt(((xx - w / 8) / (w / 8)) ** 2 + ((yy - h / 8) / (h / 8)) ** 2)
    v = 1 - 0.6 * np.clip((d - 0.75) / 0.6, 0, 1) ** 1.5
    vig = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
    vig3 = cv2.merge([(vig * 255).astype(np.uint8)] * 3)
    tint = cv2.merge([((1 - vig) * 40).astype(np.uint8), ((1 - vig) * 14).astype(np.uint8),
                      np.zeros((h, w), np.uint8)])
    refl = np.exp(-(((xx / (w / 4)) - (yy / (h / 4)) * 0.6 - 0.15) / 0.05) ** 2) * 10
    refl = cv2.resize(refl.astype(np.float32), (w, h)).astype(np.uint8)
    L = cv2.add(L, cv2.add(tint, cv2.merge([refl] * 3)))
    _cache[key] = (L, vig3, rc, rr)
    return _cache[key]


def _sfx_blast():
    if "blast" not in _cache:
        import audio as A
        n = int(A.SR * 0.55)
        t = np.arange(n) / A.SR
        thump = np.sin(2 * np.pi * np.cumsum(55 + 180 * np.exp(-t * 18)) / A.SR) * np.exp(-t * 7)
        zap = np.sin(2 * np.pi * np.cumsum(2400 * np.exp(-t * 9) + 300) / A.SR) * np.exp(-t * 14) * 0.35
        noise = np.convolve(np.random.uniform(-1, 1, n), np.ones(6) / 6, mode="same") * np.exp(-t * 10) * 0.5
        _cache["blast"] = ((thump + zap + noise) * 0.6).astype(np.float32)
    return _cache["blast"]


# =========================================================================== effet
def armure(img, c):
    """Armure high-tech : interface de casque, réacteur sur la poitrine, propulseurs dans les paumes."""
    h, w = img.shape[:2]
    S = h / 720
    t = c["t"]
    dt = min(c.get("dt", 0.016), 0.1)
    st = c["state"].setdefault("armor", {})
    if t - st.get("last", -9) > 0.5:  # effet (ré)activé → séquence de démarrage
        st.update(t0=t, charges={}, blasts=[], lock=0.0, shake=0.0, fired=0)
    st["last"] = t
    e = t - st["t0"]
    boot = min(1.0, e / 1.6)

    out = cv2.LUT(img, _grade())
    L, vig3, rc, rr = _static(w, h)

    # ---------------------------------------------------------- réacteur (poitrine)
    body, face = c.get("body"), c.get("face")
    center = rad = None
    if body is not None and min(body.vis[11], body.vis[12]) > 0.5:
        sw = float(np.linalg.norm(body.pts[11] - body.pts[12]))
        center = (body.pts[11] + body.pts[12]) / 2 + (0, 0.24 * sw)  # haut du sternum
        rad = 0.11 * sw
    elif face is not None:
        center = face.center + (0, 1.45 * face.height)
        rad = 0.24 * face.width
    if center is not None:  # sur une webcam, la poitrine est souvent au bord du cadre : on la garde visible
        center = np.array([center[0], min(center[1], h - rad * 1.25)], np.float32)
    if center is not None and center[1] < h + rad and boot > 0.3:
        pulse = 0.85 + 0.15 * math.sin(t * 5) + 0.05 * math.sin(t * 23)
        k = min(1.0, (boot - 0.3) / 0.4) * pulse
        rot = t * 25

        def reactor(p, cc, R=rad, k=k, rot=rot):
            col = tuple(int(v * k) for v in ICE)
            core = tuple(int(v * k) for v in CORE)
            cv2.circle(p, (int(cc[0]), int(cc[1])), int(R), col, max(2, int(R * 0.09)), cv2.LINE_AA)
            for i in range(10):
                a0 = math.radians(rot + i * 36 + 4)
                a1 = math.radians(rot + i * 36 + 30)
                pts = np.array([[cc[0] + math.cos(a0) * R * 0.55, cc[1] + math.sin(a0) * R * 0.55],
                                [cc[0] + math.cos(a0) * R * 0.85, cc[1] + math.sin(a0) * R * 0.85],
                                [cc[0] + math.cos(a1) * R * 0.85, cc[1] + math.sin(a1) * R * 0.85],
                                [cc[0] + math.cos(a1) * R * 0.55, cc[1] + math.sin(a1) * R * 0.55]], np.int32)
                cv2.fillConvexPoly(p, pts, col, cv2.LINE_AA)
            cv2.circle(p, (int(cc[0]), int(cc[1])), int(R * 0.42), core, -1, cv2.LINE_AA)
        _emit(out, center, rad * 4, reactor, rad * 0.9, 2.2)

    # ---------------------------------------------------------- propulseurs (paumes)
    snd = c.get("audio_obj")
    for pose, hd in c.get("hand_poses", []):
        if hd is None:
            continue
        key = hd.label
        palm = hd.pts[[0, 5, 9, 13, 17]].mean(axis=0)
        size = hd.palm
        chg = st["charges"].get(key, 0.0)
        if pose == "open":
            chg = min(1.0, chg + dt * 1.2)
        else:
            chg = max(0.0, chg - dt * 3)
        if chg >= 1.0:
            st["blasts"].append((palm.copy(), t, size))
            st["shake"] = 1.0
            st["fired"] += 1
            chg = 0.0
            if snd is not None:
                snd.play(_sfx_blast(), 0.7)
        st["charges"][key] = chg
        if chg > 0.02:
            def repulsor(p, cc, s=size, k=chg, rot=t * 240):
                col = tuple(int(v * k) for v in ICE)
                cv2.circle(p, (int(cc[0]), int(cc[1])), int(s * (0.12 + 0.2 * k)), CORE, -1, cv2.LINE_AA)
                for i in range(6):
                    a0 = rot + i * 60
                    cv2.ellipse(p, (int(cc[0]), int(cc[1])), (int(s * 0.55), int(s * 0.55)), 0, a0, a0 + 35, col,
                                max(2, int(s * 0.05)), cv2.LINE_AA)
            _emit(out, palm, size * 1.6, repulsor, size * 0.35, 2.5 * chg + 0.5)
        # repère du propulseur sur l'interface
        if boot >= 1:
            r_ = int(size * 0.85)
            cv2.circle(out, (int(palm[0]), int(palm[1])), r_, ICE, max(1, int(1.4 * S)), cv2.LINE_AA)
            ui.arc(out, palm, r_ + ui.u(6), chg, AMBER, 3)
            _txt(out, f"PROPULSEUR {int(chg * 100)}%", (palm[0] + r_ + ui.u(10), palm[1] - ui.u(10)), 13)

    # ---------------------------------------------------------- tirs
    alive = []
    for p, t0, s in st["blasts"]:
        e2 = t - t0
        if e2 > 0.7:
            continue
        alive.append((p, t0, s))
        k = 1 - e2 / 0.7

        def blast(pp, cc, s=s, e2=e2, k=k):
            cv2.circle(pp, (int(cc[0]), int(cc[1])), int(s * (0.6 + 3.5 * e2)), tuple(int(v * k) for v in CORE), -1,
                       cv2.LINE_AA)
            cv2.circle(pp, (int(cc[0]), int(cc[1])), int(s * (1 + 9 * e2)), tuple(int(v * k) for v in ICE),
                       max(2, int(s * 0.12 * k)), cv2.LINE_AA)
            # traînée anamorphique (comme un reflet d'objectif de cinéma)
            cv2.line(pp, (int(cc[0] - s * 7), int(cc[1])), (int(cc[0] + s * 7), int(cc[1])),
                     tuple(int(v * k) for v in ICE), max(2, int(s * 0.08)), cv2.LINE_AA)
        _emit(out, p, s * 8, blast, s * 1.2, 2.5)
    st["blasts"] = alive
    if st["shake"] > 0:
        st["shake"] = max(0.0, st["shake"] - dt * 4)
        out = np.roll(out, int(math.sin(t * 80) * ui.u(10) * st["shake"]), axis=1)

    # ---------------------------------------------------------- interface du casque
    out = cv2.multiply(out, vig3, scale=1 / 255)
    if boot < 1:  # démarrage : l'interface se dessine de haut en bas avec une ligne de balayage
        yb = int(h * boot)
        if yb > 0:
            out[:yb] = cv2.add(out[:yb], L[:yb])
        cv2.line(out, (0, yb), (w, yb), ICE, max(2, ui.u(2)), cv2.LINE_AA)
        if int(e * 8) % 2 == 0:
            _txt(out, "DÉMARRAGE DE L'ARMURE…", (w // 2, h // 2 + ui.u(60)), 26, ICE, "mm")
        return out
    flick = 0.92 + 0.08 * math.sin(t * 37)
    out = cv2.add(out, cv2.convertScaleAbs(L, alpha=flick))

    # compas (suit la rotation de la tête)
    yaw = face.yaw if face is not None else 0.0
    heading = (yaw * 45) % 360
    cx, ty = w // 2, int(h * 0.075)
    span = int(w * 0.22)
    for deg in range(-60, 61, 5):
        val = heading + deg
        x = int(cx + deg / 60 * span)
        tick = int((12 if round(val) % 15 == 0 else 6) * S)
        cv2.line(out, (x, ty), (x, ty + tick), ICE, max(1, int(1.2 * S)), cv2.LINE_AA)
    for deg in range(-60, 61, 1):
        val = (heading + deg) % 360
        if abs(val - round(val / 45) * 45) < 0.5:
            lab = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SO", 270: "O", 315: "NO"}[int(round(val / 45) * 45) % 360]
            _txt(out, lab, (int(cx + deg / 60 * span), ty + ui.u(16)), 13, ICE, "mt")
    cv2.line(out, (cx, ty - ui.u(6)), (cx, ty + ui.u(16)), AMBER, max(2, ui.u(2)), cv2.LINE_AA)

    # verrouillage du visage
    if face is not None:
        st["lock"] = min(1.0, st["lock"] + dt * 2.2)
        p = face.pts[:, :2]
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        k = 1 + 0.7 * (1 - ui.ease_out(st["lock"]))
        fx, fy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) / 2 * 1.25 * k, (y1 - y0) / 2 * 1.2 * k
        col = AMBER if st["lock"] >= 1 else ICE
        ln = int(min(hw, hh) * 0.35)
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            X, Y = int(fx + sx * hw), int(fy + sy * hh)
            cv2.line(out, (X, Y), (X - sx * ln, Y), col, max(2, ui.u(2)), cv2.LINE_AA)
            cv2.line(out, (X, Y), (X, Y - sy * ln), col, max(2, ui.u(2)), cv2.LINE_AA)
        dist = 0.15 * 0.9 * w / max(1.0, face.width)
        label = "CIBLE VERROUILLÉE" if st["lock"] >= 1 else "ACQUISITION…"
        ly = min(fy + hh + ui.u(8), h - ui.u(30))
        _txt(out, label, (fx - hw, ly), 14, col)
        _txt(out, f"{dist:.2f} m".replace(".", ","), (fx + hw, ly), 14, col, "rt")
    else:
        st["lock"] = max(0.0, st["lock"] - dt * 2)

    # radar : balayage + mains
    a = t * 2.2
    cv2.line(out, rc, (int(rc[0] + math.cos(a) * rr), int(rc[1] + math.sin(a) * rr)), AMBER, max(1, ui.u(1.5)),
             cv2.LINE_AA)
    for _, hd in c.get("hand_poses", []):
        if hd is None:
            continue
        bx = rc[0] + (hd.pts[9][0] / w - 0.5) * rr * 1.6
        by = rc[1] + (hd.pts[9][1] / h - 0.5) * rr * 1.6
        cv2.circle(out, (int(bx), int(by)), max(2, ui.u(4)), AMBER, -1, cv2.LINE_AA)

    # jauge de puissance (gauche)
    lvl = 0.86 + 0.08 * math.sin(t * 1.3)
    x = int(0.06 * w) + ui.u(22)
    yt, yb = int(h * 0.3), int(h * 0.7)
    cv2.rectangle(out, (x, yt), (x + ui.u(8), yb), ICE, max(1, int(S)), cv2.LINE_AA)
    cv2.rectangle(out, (x, int(yb - (yb - yt) * lvl)), (x + ui.u(8), yb), ICE, -1)
    _txt(out, f"{int(lvl * 100)}%", (x - ui.u(4), yt - ui.u(22)), 14)

    # texte système qui s'écrit
    lines = ["SYSTÈMES EN LIGNE", f"PUISSANCE {int(lvl * 100)} %", "RÉACTEUR : STABLE",
             "PROPULSEURS : " + ("TIR" if alive else "PRÊTS"), f"TIRS : {st['fired']}", time.strftime("%H:%M:%S")]
    n_chars = int((e - 1.6) * 45)
    y = int(h * 0.74)
    for ln_ in lines:
        if n_chars <= 0:
            break
        _txt(out, ln_[:n_chars], (int(0.035 * w), y), 14, ICE, alpha=0.85)
        n_chars -= len(ln_)
        y += ui.u(20)
    return out
