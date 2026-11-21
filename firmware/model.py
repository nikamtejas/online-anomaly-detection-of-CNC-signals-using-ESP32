"""Mahalanobis score on the ESP32. Must give the same score as src/models.py divided by the threshold (checked in notebook 04)."""
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


def load(op, period, path='model.bin'):
    """Read the record of this operation and period into _m. Called when the operation changes, not per window."""
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
