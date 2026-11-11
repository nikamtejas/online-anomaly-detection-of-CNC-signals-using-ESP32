"""Run firmware/ in the MicroPython Unix port and compare it with the laptop. Run from the repo root: python -m sim.run_sim"""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.features import N_PER_AXIS, features, index_recordings, load_recording

FIRMWARE = Path(__file__).parents[1] / 'firmware'
N_WINDOWS = 60  # per recording; the stand-in FFT below is slow

# emlearn_fft has no build for this Mac (arm64), so this copies fft_forward from its fft.c. Only the FFT is replaced.
FFT_STAND_IN = '''
from math import cos, pi, sin
class FFT:
    def __init__(self, n):
        self.n, self.levels = n, 0
        while 1 << self.levels < n:
            self.levels += 1
    def run(self, re, im):
        n, s, c = self.n, self.sin, self.cos
        for i in range(n):
            j = 0
            for b in range(self.levels):
                j = j << 1 | i >> b & 1
            if j > i:
                re[i], re[j] = re[j], re[i]
                im[i], im[j] = im[j], im[i]
        size = 2
        while size <= n:
            half, step = size // 2, n // size
            for i in range(0, n, size):
                k = 0
                for j in range(i, i + half):
                    l = j + half
                    tre = re[l] * c[k] + im[l] * s[k]
                    tim = -re[l] * s[k] + im[l] * c[k]
                    re[l], im[l] = re[j] - tre, im[j] - tim
                    re[j] += tre
                    im[j] += tim
                    k += step
            size *= 2
def fill(fft, n):
    fft.sin = [sin(2 * pi * i / n) for i in range(n // 2)]
    fft.cos = [cos(2 * pi * i / n) for i in range(n // 2)]
'''

# Feeds samples through a ring buffer in time order, the way they arrive on the board.
DRIVER = '''
import sys
from array import array
from features import N_AXES, N_PER_AXIS, STEP, WIN, window_features
with open(sys.argv[1], 'rb') as f:
    data = array('f', bytearray(f.read()))
rings = [array('f', bytes(4 * WIN)) for _ in range(N_AXES)]
out = array('f', bytes(4 * N_AXES * N_PER_AXIS))
with open(sys.argv[2], 'wb') as f:
    for i in range(len(data) // N_AXES):
        for a in range(N_AXES):
            rings[a][i & (WIN - 1)] = data[N_AXES * i + a]
        seen = i + 1
        if seen >= WIN and (seen - WIN) % STEP == 0:
            window_features(rings, seen & (WIN - 1), out)
            f.write(out)
'''


def run_firmware(x):
    """(n, 3) float32 signal -> features computed by firmware/features.py inside MicroPython."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'emlearn_fft.py').write_text(FFT_STAND_IN)
        (tmp / 'driver.py').write_text(DRIVER)
        x.astype('<f4').tofile(tmp / 'in.bin')
        env = {**os.environ, 'MICROPYPATH': f'{FIRMWARE}:{tmp}'}
        subprocess.run(['micropython', '-X', 'heapsize=64M', tmp / 'driver.py', tmp / 'in.bin', tmp / 'out.bin'], env=env, check=True)
        return np.fromfile(tmp / 'out.bin', '<f4').reshape(-1, 3 * N_PER_AXIS)


def compare(cases):
    """Largest difference per feature kind. RMS and peak-to-peak are in counts, the others have no unit."""
    gap = np.vstack([np.abs(run_firmware(x) - features(x)) for x in cases.values()])
    print(f'{len(cases)} signals, {len(gap)} windows. Largest absolute difference, firmware vs laptop:')
    for name, col in [('RMS (counts)', [0]), ('peak-to-peak (counts)', [1]), ('crest factor', [2]), ('log band energy', range(3, N_PER_AXIS))]:
        print(f'  {name:22} {gap[:, [a * N_PER_AXIS + c for a in range(3) for c in col]].max():.2e}')


if __name__ == '__main__':
    idx = index_recordings()
    test = idx[idx['period'].str.startswith('2021')]
    picks = test.groupby(['machine', 'label']).head(1)  # one good and one bad recording per machine
    n = 1024 + (N_WINDOWS - 1) * 256
    cases = {row['id']: load_recording(row['path'])[:n] for _, row in picks.iterrows()}
    cases['flat'] = np.zeros((1024 + 256, 3), 'float32')  # the EPS path: no vibration at all
    compare(cases)
