"""Audio : mixeur temps réel (musique + bruitages), détection des claps au micro, synthé de musique.

Le micro sert uniquement à mesurer le niveau sonore et repérer les claps : rien n'est enregistré.
Si l'audio n'est pas dispo (pas de permission, pas de module), tout continue sans son.
"""
import collections
import threading
import time

import numpy as np

try:
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None

SR = 48000


class Voice:
    def __init__(self, data, vol, loop=False):
        self.data, self.vol, self.loop = data.astype(np.float32), vol, loop
        self.pos = 0
        self.done = False


class Audio:
    def __init__(self, mic=True):
        self.ok = False
        self.mic_ok = False
        self.voices = []
        self.lock = threading.Lock()
        self.level = 0.0         # niveau micro lissé (0..1)
        self.onsets = collections.deque(maxlen=32)
        self._floor = 0.002
        self._prev = collections.deque([0.0] * 4, maxlen=4)
        self._last_onset = 0.0
        self._cand = None          # clap candidat en attente de confirmation (décroissance rapide)
        self._out_trans = 0.0      # dernier transitoire joué par nos haut-parleurs
        self._out_prev = 0.0
        self._xprev = 0.0
        self.out_latency = 0.0
        self.out = self.inp = None
        if sd is None:
            return
        try:
            self.out = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", blocksize=256,
                                       callback=self._out_cb, latency="low")
            self.out.start()
            self.out_latency = float(self.out.latency)
            self.ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[audio] sortie indisponible : {e}")
        if mic:
            try:
                self.inp = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=256,
                                          callback=self._in_cb, latency="low")
                self.inp.start()
                self.mic_ok = True
            except Exception as e:  # noqa: BLE001
                print(f"[audio] micro indisponible : {e}")

    # ------------------------------------------------------------------ sortie
    def _out_cb(self, outdata, frames, _t, _status):
        buf = np.zeros(frames, np.float32)
        with self.lock:
            for v in self.voices:
                n = min(frames, len(v.data) - v.pos)
                if n > 0:
                    buf[:n] += v.data[v.pos:v.pos + n] * v.vol
                v.pos += frames
                if v.pos >= len(v.data):
                    if v.loop:
                        v.pos = 0
                    else:
                        v.done = True
            self.voices = [v for v in self.voices if not v.done]
        out = np.tanh(buf * 0.9)  # limiteur doux
        outdata[:, 0] = out
        # on note nos propres "claquements" (musique, bruitages) pour ne pas les prendre pour des claps
        hf = float(np.sqrt(np.mean(np.diff(out) ** 2)) + 1e-7)
        if hf > 0.02 and hf > 2.5 * self._out_prev:
            self._out_trans = time.monotonic()
        self._out_prev = 0.7 * self._out_prev + 0.3 * hf

    def play(self, data, vol=1.0, loop=False):
        v = Voice(data, vol, loop)
        if self.ok:
            with self.lock:
                self.voices.append(v)
        return v

    def stop(self, voice):
        if voice is None:
            return
        with self.lock:
            voice.done = True
            self.voices = [v for v in self.voices if v is not voice]

    def voice_time(self, voice):
        """Temps de lecture réellement entendu (en s), latence de sortie déduite."""
        return voice.pos / SR - self.out_latency

    # ------------------------------------------------------------------ micro
    def _in_cb(self, indata, frames, _t, _status):
        """Détecteur de clap :
        1. énergie des aigus (un clap est un bruit sec, riche en aigus ; la voix et les basses beaucoup moins)
        2. montée brutale par rapport au bruit de fond ET aux blocs précédents
        3. confirmation ~50 ms plus tard : l'énergie doit être retombée (un clap est très bref)
        4. ignoré si nos haut-parleurs viennent de jouer un transitoire (musique du jeu, bruitages)"""
        x = indata[:, 0].astype(np.float32)
        now = time.monotonic()
        rms = float(np.sqrt(np.mean(x * x)) + 1e-7)
        self.level = max(rms * 6, self.level * 0.92)
        y = np.diff(np.concatenate([[self._xprev], x]))  # passe-haut (dérivée) → aigus
        self._xprev = float(x[-1])
        hf = float(np.sqrt(np.mean(y * y)) + 1e-7)
        hist = list(self._prev)
        prev = max(hist[:-1])  # on ignore le bloc juste avant (le début du clap peut y tomber à moitié)
        self._nblk = getattr(self, "_nblk", 0) + 1
        if self._cand is None:
            if (hf > 0.012 and hf > 12 * self._floor and hf > 5 * prev and hf > 1.3 * hist[-1]
                    and now - self._last_onset > 0.12 and now - self._out_trans > 0.15):
                self._cand = (now, hf, 0)
            else:
                k = 0.1 if self._nblk < 40 else 0.01  # adaptation rapide au démarrage
                self._floor = (1 - k) * self._floor + k * min(hf, self._floor * 2.5 if self._nblk > 40 else hf)
        else:
            t0, peak, n = self._cand
            peak = max(peak, hf)
            n += 1
            if n >= 9:  # ~50 ms après
                if hf < 0.3 * peak:
                    self.onsets.append(t0)
                    self._last_onset = t0
                self._cand = None
            else:
                self._cand = (t0, peak, n)
        self._prev.append(hf)

    def onset_since(self, t):
        return any(o > t for o in list(self.onsets))

    def close(self):
        for s in (self.out, self.inp):
            try:
                if s is not None:
                    s.stop()
                    s.close()
            except Exception:  # noqa: BLE001
                pass


# =========================================================================== synthé
def _env(n, a=0.002, r=0.1):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / max(r, 1e-4))


def sfx_pop(pitch=1.0):
    n = int(SR * 0.12)
    t = np.arange(n) / SR
    f = 900 * pitch * np.exp(-t * 18) + 300
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.001, 0.05) * 0.6


def sfx_coin():
    n = int(SR * 0.25)
    t = np.arange(n) / SR
    f = np.where(t < 0.07, 988, 1319)
    return np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * _env(n, 0.001, 0.12) * 0.25


def sfx_bad():
    n = int(SR * 0.35)
    t = np.arange(n) / SR
    f = 220 * np.exp(-t * 3)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.5 + np.random.uniform(-1, 1, n) * 0.25) * _env(n, 0.002, 0.15)


def sfx_chime(notes=(523, 659, 784, 1047)):
    out = np.zeros(int(SR * 0.9), np.float32)
    for i, f in enumerate(notes):
        n = int(SR * 0.6)
        s = int(i * SR * 0.07)
        t = np.arange(n) / SR
        tone = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * _env(n, 0.003, 0.25) * 0.3
        out[s:s + n] += tone[:len(out) - s]
    return out


def sfx_whoosh():
    n = int(SR * 0.25)
    noise = np.random.uniform(-1, 1, n)
    k = np.exp(-((np.arange(n) - n * 0.4) / (n * 0.18)) ** 2)
    return np.convolve(noise, np.ones(24) / 24, mode="same") * k * 0.8


SONGS = [
    {"name": "Néon Pop", "bpm": 118, "prog": [(57, [57, 60, 64]), (53, [53, 57, 60]), (48, [48, 52, 55]),
                                              (55, [55, 59, 62])], "style": "pop", "seed": 4},
    {"name": "Boule à facettes", "bpm": 110, "prog": [(50, [50, 53, 57, 60]), (55, [55, 59, 62, 65]),
                                                      (48, [48, 52, 55, 59]), (57, [57, 60, 64, 67])],
     "style": "funk", "seed": 7},
    {"name": "Turbo", "bpm": 128, "prog": [(53, [53, 56, 60]), (49, [49, 53, 56]), (56, [56, 60, 63]),
                                           (51, [51, 55, 58])], "style": "house", "seed": 11},
]


def make_track(bpm=118, bars=26, seed=4, prog=None, style="pop"):
    """Petit morceau synthétisé. Styles : pop (four-on-the-floor + arpège), funk (basse syncopée,
    accords en contretemps), house (basse en contretemps, nappes, arpège rapide)."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n = int(SR * beat * 4 * bars) + SR
    out = np.zeros(n, np.float32)
    t_beat = int(SR * beat)
    prog = prog or SONGS[0]["prog"]

    def midi(m):
        return 440.0 * 2 ** ((m - 69) / 12)

    def add(sig, start, vol=1.0):
        s = int(start)
        e = min(n, s + len(sig))
        if e > s:
            out[s:e] += sig[:e - s] * vol

    kn = int(SR * 0.35)
    tk = np.arange(kn) / SR
    kick = np.sin(2 * np.pi * np.cumsum(50 + 120 * np.exp(-tk * 30)) / SR) * _env(kn, 0.001, 0.12)
    cn = int(SR * 0.2)
    clap = np.random.uniform(-1, 1, cn) * _env(cn, 0.001, 0.05)
    hn = int(SR * 0.05)
    hat = np.diff(np.random.uniform(-1, 1, hn + 1)) * _env(hn, 0.0005, 0.015)
    on = int(SR * 0.12)
    ohat = np.diff(np.random.uniform(-1, 1, on + 1)) * _env(on, 0.001, 0.05)

    def bass_note(f, dur, vol):
        bn = int(dur)
        tb = np.arange(bn) / SR
        sig = (2 * ((tb * f) % 1) - 1) * _env(bn, 0.003, 0.12)
        return np.convolve(sig, np.ones(8) / 8, mode="same") * vol

    for bar in range(bars):
        root, chord = prog[bar % len(prog)]
        intro = bar < 2
        outro = bar >= bars - 1
        bs = bar * 4 * t_beat
        for b in range(4):
            st = bs + b * t_beat
            if not intro:
                if style == "funk":
                    if b in (0, 2):
                        add(kick, st, 0.9)
                    if b == 2:
                        add(kick, st + t_beat * 3 // 4, 0.6)
                else:
                    add(kick, st, 0.9)
            if b in (1, 3) and not intro:
                add(clap, st, 0.35)
            for half in (0, 1):
                if style == "house" and half:
                    add(ohat, st + t_beat // 2, 0.12)
                else:
                    add(hat, st + half * t_beat // 2, 0.12 if half else 0.06)
            if intro or outro:
                continue
            if style == "pop":
                for e in (0, 1):
                    add(bass_note(midi(root - 12 + 12 * e), t_beat // 2, 0.35), st + e * t_beat // 2)
            elif style == "house":
                add(bass_note(midi(root - 12), t_beat // 2, 0.4), st + t_beat // 2)
            else:  # funk : basse syncopée en octaves
                pat = [(0, 0), (3, 12), (6, 0), (7, 10)]
                for s16, iv in pat:
                    if rng.random() < 0.85:
                        add(bass_note(midi(root - 12 + iv), t_beat // 4, 0.4), st + s16 * t_beat // 8)
        # nappe / accords
        pn = t_beat * 4
        tp = np.arange(pn) / SR
        if style == "funk" and not intro:
            for b in range(4):
                sn = t_beat // 3
                ts_ = np.arange(sn) / SR
                stab = sum(np.sign(np.sin(2 * np.pi * midi(m + 12) * ts_)) for m in chord) * _env(sn, 0.002, 0.05)
                add(stab, bs + b * t_beat + t_beat // 2, 0.03)
        else:
            pad = sum(np.sin(2 * np.pi * midi(m) * tp) + 0.5 * np.sin(2 * np.pi * midi(m) * 1.004 * tp) for m in chord)
            add(pad * np.minimum(1, tp / 0.3) * np.minimum(1, (pn / SR - tp) / 0.3) * 0.05, bs)
        # arpège
        if bar >= 4 and not outro and style != "funk":
            steps = 16 if style == "pop" else 32
            for s in range(steps):
                m = chord[rng.integers(0, len(chord))] + 12 * rng.integers(1, 3)
                an = t_beat * 4 // steps
                ta = np.arange(an) / SR
                tone = np.sign(np.sin(2 * np.pi * midi(m) * ta)) * _env(an, 0.002, 0.05)
                add(tone, bs + s * an, 0.045)
    out /= max(1e-6, np.max(np.abs(out))) / 0.8
    return out.astype(np.float32), bpm


def sfx_shutter():
    n = int(SR * 0.12)
    noise = np.random.uniform(-1, 1, n)
    env = np.exp(-np.arange(n) / SR / 0.012) + 0.6 * np.exp(-np.maximum(0, np.arange(n) / SR - 0.06) / 0.01) * (
        np.arange(n) / SR > 0.06)
    return (noise * env * 0.5).astype(np.float32)


def sfx_boom():
    n = int(SR * 0.6)
    t = np.arange(n) / SR
    low = np.sin(2 * np.pi * np.cumsum(40 + 80 * np.exp(-t * 8)) / SR)
    return ((low * 0.7 + np.random.uniform(-1, 1, n) * 0.4) * _env(n, 0.002, 0.2)).astype(np.float32)


def sfx_slice():
    n = int(SR * 0.15)
    noise = np.random.uniform(-1, 1, n)
    hp = np.diff(np.concatenate([[0], noise]))
    k = np.exp(-((np.arange(n) - n * 0.3) / (n * 0.15)) ** 2)
    return (hp * k * 0.5).astype(np.float32)
