"""Main loop on the ESP32: samples over WiFi, one window every 256 samples, score, LED, one serial line per window."""
import gc
import socket
import time
from array import array

import network
from machine import Pin
from neopixel import NeoPixel

import model
from features import N_AXES, N_FEAT, STEP, WIN, spindle_bins, window_features
from secrets import PASSWORD, SSID

PORT = 5005
LED_PIN = 48  # UNVERIFIED: onboard RGB LED is GPIO48 on DevKitC-1 v1.0, GPIO38 on v1.1
RED, OFF = (32, 0, 0), (0, 0, 0)

# Per recording, tools/stream.py sends a 13-byte header 'OP01 2019-08\n', then X, Y, Z as little-endian int16 per sample.
HEAD = 13

# All buffers are allocated once, here.
led = NeoPixel(Pin(LED_PIN), 1)
head = bytearray(HEAD)
chunk = array('h', bytes(2 * N_AXES * STEP))  # the next 256 samples, as they arrive
rings = [array('f', bytes(4 * WIN)) for _ in range(N_AXES)]
feat = array('f', bytes(4 * N_FEAT))


def show(on):
    led[0] = RED if on else OFF
    led.write()


def join_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(SSID, PASSWORD)
    while not wlan.isconnected():
        time.sleep_ms(100)
    print('ip', wlan.ifconfig()[0])


def run(conn):
    """One recording: header, then a window every STEP samples until the laptop closes the connection."""
    if conn.readinto(head) < HEAD:
        return
    op, period = bytes(head[:4]).decode(), bytes(head[5:12]).decode()
    try:
        model.load(op, period)
    except KeyError:
        print('no model for', op, period)
        return
    band = spindle_bins(op)
    seen = 0
    while conn.readinto(chunk) == 2 * len(chunk):  # readinto waits for all 256 samples, less only at the end
        t = time.ticks_us()
        for i in range(STEP):
            j = (seen + i) & (WIN - 1)
            for a in range(N_AXES):
                rings[a][j] = chunk[N_AXES * i + a]
        seen += STEP
        if seen < WIN:  # first full window after 1024 samples
            continue
        window_features(rings, seen & (WIN - 1), band, feat)
        on = model.update(feat)
        show(on)
        dt = time.ticks_diff(time.ticks_us(), t)  # time on the chip only, without waiting for WiFi
        print('%.6f,%d,%d' % (model.now, on, dt))
        gc.collect()


join_wifi()
show(False)
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(socket.getaddrinfo('0.0.0.0', PORT)[0][-1])
server.listen(1)
gc.collect()
print('mem_free', gc.mem_free())
while True:
    conn, _ = server.accept()
    run(conn)
    conn.close()
    show(False)
    print('done')
