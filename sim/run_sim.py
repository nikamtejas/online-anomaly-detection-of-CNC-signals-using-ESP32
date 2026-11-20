"""Run firmware/ in the MicroPython Unix port, so notebook 04 can compare it with the laptop."""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

FIRMWARE = Path(__file__).parents[1] / 'firmware'

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
from features import N_AXES, N_FEAT, STEP, WIN, spindle_bins, window_features
with open(sys.argv[1], 'rb') as f:
    data = array('f', f.read())
band = spindle_bins(sys.argv[3])
rings = [array('f', bytes(4 * WIN)) for _ in range(N_AXES)]
out = array('f', bytes(4 * N_FEAT))
with open(sys.argv[2], 'wb') as f:
    for i in range(len(data) // N_AXES):
        for a in range(N_AXES):
            rings[a][i & (WIN - 1)] = data[N_AXES * i + a]
        seen = i + 1
        if seen >= WIN and (seen - WIN) % STEP == 0:
            window_features(rings, seen & (WIN - 1), band, out)
            f.write(out)
'''


def run_firmware(x, op):
    """(n, 3) signal in counts -> (n_windows, 12) features computed by firmware/features.py in MicroPython."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'emlearn_fft.py').write_text(FFT_STAND_IN)
        (tmp / 'driver.py').write_text(DRIVER)
        np.asarray(x, '<f4').tofile(tmp / 'in.bin')
        env = {**os.environ, 'MICROPYPATH': f'{FIRMWARE}:{tmp}'}
        subprocess.run(['micropython', '-X', 'heapsize=64M', tmp / 'driver.py', tmp / 'in.bin', tmp / 'out.bin', op],
                       env=env, check=True)
        return np.fromfile(tmp / 'out.bin', '<f4').reshape(-1, 12)
