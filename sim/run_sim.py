"""Run firmware/ in the MicroPython Unix port, so notebook 04 can compare it with the laptop."""
import os
import socket
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
# With a period as 4th argument it also loads model.bin and writes score, 5 s median and LED after the 12 features.
DRIVER = '''
import sys
from array import array
from features import N_AXES, N_FEAT, STEP, WIN, spindle_bins, window_features
with open(sys.argv[1], 'rb') as f:
    data = array('f', f.read())
band = spindle_bins(sys.argv[3])
scoring = len(sys.argv) > 4
if scoring:
    import model
    model.load(sys.argv[3], sys.argv[4])
rings = [array('f', bytes(4 * WIN)) for _ in range(N_AXES)]
out = array('f', bytes(4 * N_FEAT))
s = array('f', [0, 0, 0])
with open(sys.argv[2], 'wb') as f:
    for i in range(len(data) // N_AXES):
        for a in range(N_AXES):
            rings[a][i & (WIN - 1)] = data[N_AXES * i + a]
        seen = i + 1
        if seen >= WIN and (seen - WIN) % STEP == 0:
            window_features(rings, seen & (WIN - 1), band, out)
            f.write(out)
            if scoring:
                s[2] = model.update(out)
                s[0], s[1] = model.now, model.med
                f.write(s)
'''


# Stand-ins for the board-only modules main.py imports. WiFi is "connected" at once and the LED does nothing,
# so main.py runs unchanged and serves on 127.0.0.1. Its serial line already says whether the LED is on.
BOARD_STAND_INS = {
    'network.py': '''
STA_IF = 0
class WLAN:
    def __init__(self, i): pass
    def active(self, on): pass
    def connect(self, ssid, password): pass
    def isconnected(self): return True
    def ifconfig(self): return ('127.0.0.1', '', '', '')
''',
    'machine.py': '''
class Pin:
    def __init__(self, n): pass
''',
    'neopixel.py': '''
class NeoPixel:
    def __init__(self, pin, n): self.c = [(0, 0, 0)] * n
    def __setitem__(self, i, c): self.c[i] = c
    def write(self): pass
''',
    'secrets.py': "SSID = PASSWORD = ''\n",
}
PORT = 5005  # same as firmware/main.py


def _run(x, args, cols):
    """Start the driver in MicroPython with firmware/ as working directory, like / on the board."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'emlearn_fft.py').write_text(FFT_STAND_IN)
        (tmp / 'driver.py').write_text(DRIVER)
        np.asarray(x, '<f4').tofile(tmp / 'in.bin')
        env = {**os.environ, 'MICROPYPATH': f'{FIRMWARE}:{tmp}'}
        subprocess.run(['micropython', '-X', 'heapsize=64M', tmp / 'driver.py', tmp / 'in.bin', tmp / 'out.bin', *args],
                       env=env, cwd=FIRMWARE, check=True)
        return np.fromfile(tmp / 'out.bin', '<f4').reshape(-1, cols)


def run_firmware(x, op):
    """(n, 3) signal in counts -> (n_windows, 12) features computed by firmware/features.py in MicroPython."""
    return _run(x, [op], 12)


def run_scores(x, op, period):
    """(n, 3) signal in counts -> (n_windows,) score / threshold from firmware/model.py, with firmware/model.bin."""
    return _run(x, [op, period], 15)[:, 12]


def run_alarm(x, op, period):
    """(n, 3) signal in counts -> 5 s median score and LED (True/False) per window, from firmware/model.py."""
    out = _run(x, [op, period], 15)
    return out[:, 13], out[:, 14] > 0


def run_main(recordings, piece=None):
    """Start firmware/main.py and send it each (x, op, period) over TCP, like tools/stream.py.
    Returns one (n_windows, 3) array per recording: score / threshold, LED, latency in us. piece: send in pieces of this many bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'emlearn_fft.py').write_text(FFT_STAND_IN)
        for name, code in BOARD_STAND_INS.items():
            (tmp / name).write_text(code)
        # secrets.py is gitignored, so the stand-in goes first in the path and the real one is never read on the laptop
        env = {**os.environ, 'MICROPYPATH': f'{tmp}:{FIRMWARE}'}
        p = subprocess.Popen(['micropython', FIRMWARE / 'main.py'], env=env, cwd=FIRMWARE, stdout=subprocess.PIPE, text=True)
        try:
            while not p.stdout.readline().startswith('mem_free'):
                pass
            outs = []
            for x, op, period in recordings:
                data = f'{op} {period}\n'.encode() + np.asarray(x, '<i2').tobytes()
                with socket.create_connection(('127.0.0.1', PORT)) as c:
                    for k in range(0, len(data), piece or len(data)):
                        c.sendall(data[k:k + (piece or len(data))])
                rows = []
                for line in p.stdout:
                    if line.startswith('done'):
                        break
                    rows.append([float(v) for v in line.split(',')])
                outs.append(np.array(rows).reshape(-1, 3))
            return outs
        finally:
            p.kill()
            p.wait()
