"""Mahalanobis score and LED rule on the ESP32. Must match src/models.py and notebook 03 (checked in notebook 04)."""
from array import array
from math import sqrt

from features import N_FEAT

# model.bin: one record per operation and period, written by notebook 04.
# Record = 12-byte key 'OP01 2019-08', then float32: centre (12), std (12), inverse covariance (12 x 12), threshold (1).
KEY = 12
SIZE = N_FEAT + N_FEAT + N_FEAT * N_FEAT + 1
STD, INV, THR = N_FEAT, 2 * N_FEAT, 2 * N_FEAT + N_FEAT * N_FEAT  # where each part starts

_key = bytearray(KEY)
_m = array('f', bytes(4 * SIZE))  # the record in use
_z = array('f', bytes(4 * N_FEAT))

IDLE = 30   # counts, largest axis RMS below this: the machine stands still (src/features.IDLE)
N_MED = 40  # windows in the 5 s median
HOLD = 24   # windows the median must stay above 1 before the LED turns on (notebook 03, Day 7)
_last = array('f', bytes(4 * N_MED))  # last N_MED scores, as a ring
_sort = array('f', bytes(4 * N_MED))  # sorted copy for the median
_n = 0                                # running windows since load()
_above = 0                            # median above 1 this many windows in a row
now = med = float('nan')              # last score and last median, for the serial output


def load(op, period, path='model.bin'):
    """Read the record of this operation and period into _m and start a new run. Called when the operation changes."""
    global _n, _above, now, med
    _n = _above = 0
    now = med = float('nan')
    want = (op + ' ' + period).encode()
    with open(path, 'rb') as f:
        while f.readinto(_key) == KEY:
            f.readinto(_m)
            if _key == want:
                return
    raise KeyError(want)


def score(x):
    """Mahalanobis distance of the 12 features x from the period centre, divided by the threshold (1 = threshold)."""
    for i in range(N_FEAT):
        _z[i] = (x[i] - _m[i]) / _m[STD + i]
    d = 0.0
    for i in range(N_FEAT):
        row = INV + N_FEAT * i
        s = 0.0
        for j in range(N_FEAT):
            s += _m[row + j] * _z[j]
        d += _z[i] * s
    return sqrt(max(d, 0.0)) / _m[THR]  # max: rounding can push d a hair below 0


def median():
    """Median of the last N_MED scores: insertion sort of a copy, then the mean of the two middle values."""
    for i in range(N_MED):
        v = _last[i]
        j = i
        while j > 0 and _sort[j - 1] > v:
            _sort[j] = _sort[j - 1]
            j -= 1
        _sort[j] = v
    return (_sort[N_MED // 2 - 1] + _sort[N_MED // 2]) / 2


def update(x):
    """One window: score, 5 s median and hold. Returns True while the LED is on."""
    global _n, _above, now, med
    now = score(x)
    if max(x[0], x[1], x[2]) < IDLE:  # idle window: not in the median, LED unchanged
        return _above >= HOLD
    _last[_n % N_MED] = now
    _n += 1
    if _n < N_MED:  # median not ready yet, counts as below
        return False
    med = median()
    _above = _above + 1 if med > 1 else 0
    return _above >= HOLD
