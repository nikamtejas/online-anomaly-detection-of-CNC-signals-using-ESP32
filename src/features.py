"""Load M01 recordings and compute window features (from notebooks/02_features.ipynb)."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

FS = 2000                 # Hz, README line 15
AXES = 'XYZ'              # README line 37
WIN, STEP = 1024, 256     # 512 ms window, new result every 128 ms
SPEED = dict(OP01=250, OP02=200, OP03=250, OP04=250, OP05=200, OP06=250, OP07=200,
             OP08=250, OP09=250, OP10=250, OP11=250, OP12=250, OP14=250)  # Hz, paper Table 1
IDLE = 30                 # counts, largest axis RMS: dip between idle and running (notebook 02)
DIR = Path(__file__).parents[1] / 'data' / 'CNC_Machining' / 'data' / 'M01'

EPS = 1e-6
HANN = np.hanning(WIN)
FREQ = np.fft.rfftfreq(WIN, 1 / FS)  # 1.95 Hz per FFT bin
NAMES = [f'{k} {ax}' for k in ['rms', 'p2p', 'crest', 'spindle_db'] for ax in AXES]


def load(path):
    """(n, 3) array in counts, columns X, Y, Z."""
    with h5py.File(path) as f:
        return f['vibration_data'][()].astype(float)


def index_recordings():
    """One row per M01 recording of the operations in SPEED (OP00 and OP13 have no bad)."""
    rows = []
    for p in sorted(DIR.glob('OP*/*/*.h5')):
        _, month, year, op, num = p.stem.split('_')
        if op in SPEED:
            rows.append(dict(op=op, period=pd.to_datetime(month + year, format='%b%Y').strftime('%Y-%m'),
                             label=p.parent.name, num=num, path=p))
    return pd.DataFrame(rows)


def windows(x):
    """(n, 3) signal -> (n_windows, 3, WIN), mean removed per window and axis."""
    w = np.lib.stride_tricks.sliding_window_view(x, WIN, axis=0)[::STEP]
    return w - w.mean(axis=2, keepdims=True)


def features(x, op):
    """(n, 3) signal -> (n_windows, 12), ordered as NAMES."""
    w = windows(x)
    rms = np.sqrt((w ** 2).mean(axis=2))
    p2p = w.max(axis=2) - w.min(axis=2)
    crest = np.abs(w).max(axis=2) / (rms + EPS)
    power = np.abs(np.fft.rfft(w * HANN, axis=2)) ** 2
    near = np.abs(FREQ - SPEED[op]) <= .1 * SPEED[op]
    spindle = 10 * np.log10(power[:, :, near].sum(axis=2) + EPS)
    return np.hstack([rms, p2p, crest, spindle])


def is_idle(F):
    """True for windows whose largest axis RMS is below IDLE."""
    return F[:, :3].max(axis=1) < IDLE
