"""Window features for the chip. Must match src/features.py, which sim/run_sim.py tests."""
from array import array
from math import cos, log, pi, sqrt

import emlearn_fft

WIN, STEP = 1024, 256  # 512 ms window, new result every 128 ms at 2 kHz
N_AXES, N_PER_AXIS = 3, 11  # RMS, peak-to-peak, crest factor, 8 bands
BAND_BINS = (5, 9, 16, 29, 51, 91, 162, 288, 513)  # same integers as src/features.py
EPS = 1e-6  # avoids log(0) and division by zero on a flat signal
HANN = array('f', (0.5 - 0.5 * cos(2 * pi * i / (WIN - 1)) for i in range(WIN)))  # same formula as np.hanning

_fft = emlearn_fft.FFT(WIN)
emlearn_fft.fill(_fft, WIN)
_re, _im = array('f', bytes(4 * WIN)), array('f', bytes(4 * WIN))  # one FFT buffer, reused for every axis


def axis_features(ring, head, out, at):
    """Write one axis to out[at:]. ring holds the last WIN samples and the oldest sits at head."""
    mean = sum(ring) / WIN  # a constant offset (possibly gravity) is not vibration
    energy, lo, hi = 0.0, 0.0, 0.0
    # One loop for all time-domain work, because every pass over 1024 samples costs time on the chip.
    for i in range(WIN):
        d = ring[(head + i) & (WIN - 1)] - mean  # WIN is a power of 2, so & wraps the ring
        energy += d * d
        if d < lo:
            lo = d
        elif d > hi:
            hi = d
        _re[i] = d * HANN[i]
        _im[i] = 0.0
    rms = sqrt(energy / WIN)
    out[at] = rms
    out[at + 1] = hi - lo
    out[at + 2] = max(hi, -lo) / (rms + EPS)
    _fft.run(_re, _im)
    for b in range(len(BAND_BINS) - 1):
        power = 0.0
        for k in range(BAND_BINS[b], BAND_BINS[b + 1]):
            power += _re[k] * _re[k] + _im[k] * _im[k]
        out[at + 3 + b] = log(power + EPS)


def window_features(rings, head, out):
    """Fill out with all features, ordered X, then Y, then Z, as on the laptop."""
    for a in range(N_AXES):
        axis_features(rings[a], head, out, a * N_PER_AXIS)
