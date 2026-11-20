"""Window features on the ESP32. Must give the same 12 numbers as src/features.py (checked in notebook 04)."""
from array import array
from math import cos, log10, pi, sqrt

import emlearn_fft

FS = 2000                 # Hz
WIN, STEP = 1024, 256     # 512 ms window, new result every 128 ms
N_AXES = 3                # X, Y, Z
N_FEAT = 12               # rms XYZ, p2p XYZ, crest XYZ, spindle_db XYZ (same order as src/features.NAMES)
SPEED = dict(OP01=250, OP02=200, OP03=250, OP04=250, OP05=200, OP06=250, OP07=200,
             OP08=250, OP09=250, OP10=250, OP11=250, OP12=250, OP14=250)  # Hz, paper Table 1
EPS = 1e-6
HANN = array('f', (0.5 - 0.5 * cos(2 * pi * i / (WIN - 1)) for i in range(WIN)))  # same formula as np.hanning

_fft = emlearn_fft.FFT(WIN)
emlearn_fft.fill(_fft, WIN)
_re, _im = array('f', bytes(4 * WIN)), array('f', bytes(4 * WIN))  # one FFT buffer, reused for every axis


def spindle_bins(op):
    """First and last FFT bin within +-10% of the operation's spindle speed."""
    s = SPEED[op]
    lo = (9 * s * WIN + 10 * FS - 1) // (10 * FS)  # ceil(0.9 * s / bin width)
    hi = 11 * s * WIN // (10 * FS)                 # floor(1.1 * s / bin width)
    return lo, hi


def axis_features(ring, head, band, out, a):
    """Features of one axis into out[a], out[3 + a], ... ring holds the last WIN samples, the oldest at head."""
    mean = 0.0
    for v in ring:
        mean += v
    mean /= WIN
    energy, lo, hi = 0.0, 0.0, 0.0
    for i in range(WIN):  # one pass for all time-domain work
        d = ring[(head + i) & (WIN - 1)] - mean  # WIN is a power of 2, so & wraps the ring
        energy += d * d
        if d < lo:
            lo = d
        elif d > hi:
            hi = d
        _re[i] = d * HANN[i]
        _im[i] = 0.0
    rms = sqrt(energy / WIN)
    out[a] = rms
    out[3 + a] = hi - lo
    out[6 + a] = max(hi, -lo) / (rms + EPS)
    _fft.run(_re, _im)
    power = 0.0
    for k in range(band[0], band[1] + 1):
        power += _re[k] * _re[k] + _im[k] * _im[k]
    out[9 + a] = 10 * log10(power + EPS)


def window_features(rings, head, band, out):
    """All 12 features of the current window into out."""
    for a in range(N_AXES):
        axis_features(rings[a], head, band, out, a)
