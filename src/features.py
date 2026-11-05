"""Load recordings and compute window features. firmware/features.py must give the same result."""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

FS = 2000  # sampling rate in Hz: dataset README line 15. The h5 files have no timestamps to confirm it.
WIN, STEP = 1024, 256  # 512 ms window, new result every 128 ms
AXES = 'XYZ'  # column order: dataset README line 37
DATA_DIR = Path(__file__).parents[1] / 'data' / 'CNC_Machining' / 'data'

N_BANDS = 8
N_PER_AXIS = 3 + N_BANDS  # RMS, peak-to-peak, crest factor, bands
# Log-spaced FFT bins from 10 Hz to 1 kHz. Below 10 Hz a 512 ms window holds only about 5 cycles.
# Whole bins, so the chip can use the same integers. +1 keeps the last bin (exactly 1 kHz).
BAND_BINS = np.round(np.geomspace(10, FS / 2, N_BANDS + 1) * WIN / FS).astype(int)
BAND_BINS[-1] += 1
HANN = np.hanning(WIN).astype('float32')
EPS = 1e-6  # avoids log(0) and division by zero on a flat signal


def index_recordings(root=DATA_DIR):
    """One row per recording: metadata from the path, length from the h5 header (no data read)."""
    rows = []
    for path in sorted(Path(root).glob('M*/OP*/*/*.h5')):
        machine, month, year, op, _ = path.stem.split('_')
        with h5py.File(path) as f:
            n, dtype = f['vibration_data'].shape[0], f['vibration_data'].dtype.name
        rows.append(dict(
            id=f'{path.stem}_{path.parent.name}',  # example numbers repeat across good/bad, so add the label
            machine=machine, op=op, label=path.parent.name,
            period=pd.to_datetime(month + year, format='%b%Y').strftime('%Y-%m'),  # sorts in time order
            n=n, dtype=dtype, path=str(path)))
    return pd.DataFrame(rows)


def load_recording(path):
    """Return an (n, 3) float32 array. Files mix int64/float32/float64 but all hold whole numbers."""
    with h5py.File(path) as f:
        return f['vibration_data'][()].astype('float32')


def features(x):
    """(n, 3) signal -> (windows, 3 * N_PER_AXIS) float32 features, ordered X, then Y, then Z."""
    w = np.lib.stride_tricks.sliding_window_view(x, WIN, axis=0)[::STEP]  # windows, axes, samples
    w = w - w.mean(axis=2, keepdims=True)  # a constant offset (possibly gravity) is not vibration
    rms = np.sqrt((w ** 2).mean(axis=2))
    power = np.abs(np.fft.rfft(w * HANN, axis=2)) ** 2
    bands = [np.log(power[:, :, a:b].sum(axis=2) + EPS) for a, b in zip(BAND_BINS[:-1], BAND_BINS[1:])]
    return np.stack([rms, np.ptp(w, axis=2), np.abs(w).max(axis=2) / (rms + EPS), *bands], axis=2).reshape(len(w), -1)
