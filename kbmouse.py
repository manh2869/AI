#!/usr/bin/env python3
"""
kbmouse.py - Dieu khien chuot bang ban phim tren Ubuntu (X11 + Wayland).

Cai dat:   sudo apt install python3-evdev
Chay:      sudo python3 kbmouse.py

Dung:
  Right Ctrl ......... bat / tat che do chuot
  Khi bat:
    Mui ten .......... di chuyen, giu cang lau cang nhanh
    Shift ............ giu de di chuyen cham (chinh xac)
    Z ................ chuot trai (giu de keo tha)
    X ................ chuot phai
    C ................ chuot giua
    A / S ............ cuon len / xuong
  Khi che do chuot bat, ban phim bi "nuot" (khong go chu ra ngoai).
"""
import select
import sys
import time

from evdev import InputDevice, UInput, ecodes as e, list_devices

# ---------- Cau hinh ----------
TOGGLE_KEY = e.KEY_RIGHTCTRL
MOVE_KEYS = {
    e.KEY_UP: (0, -1),
    e.KEY_DOWN: (0, 1),
    e.KEY_LEFT: (-1, 0),
    e.KEY_RIGHT: (1, 0),
}
BUTTON_KEYS = {
    e.KEY_Z: e.BTN_LEFT,
    e.KEY_X: e.BTN_RIGHT,
    e.KEY_C: e.BTN_MIDDLE,
}
SCROLL_KEYS = {e.KEY_A: 1, e.KEY_S: -1}  # +1 = cuon len

BASE_SPEED = 300      # pixel/giay luc moi bat dau
MAX_SPEED = 2200      # pixel/giay toi da
ACCEL_TIME = 0.8      # giay de dat toc do toi da
SLOW_FACTOR = 0.15    # he so khi giu Shift
TICK = 1 / 120        # chu ky cap nhat
# ------------------------------

VIRTUAL_NAME = "kbmouse-virtual"


def find_keyboards():
    kbs = []
    for path in list_devices():
        try:
            d = InputDevice(path)
        except OSError:
            continue
        keys = d.capabilities().get(e.EV_KEY, [])
        if e.KEY_A in keys and e.KEY_ENTER in keys and d.name != VIRTUAL_NAME:
            kbs.append(d)
    return kbs


def main():
    keyboards = find_keyboards()
    if not keyboards:
        sys.exit("Khong tim thay ban phim nao (can chay bang sudo?).")

    ui = UInput(
        {
            e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
            e.EV_REL: [e.REL_X, e.REL_Y, e.REL_WHEEL],
        },
        name=VIRTUAL_NAME,
    )

    print("Ban phim:", ", ".join(k.name for k in keyboards))
    print("Nhan Right Ctrl de bat/tat che do chuot. Ctrl+C de thoat.")

    active = False
    held_move = set()
    held_buttons = set()
    shift = False
    hold_start = None
    rem_x = rem_y = 0.0
    last = time.monotonic()

    def set_active(on):
        nonlocal active, hold_start
        active = on
        for k in keyboards:
            try:
                k.grab() if on else k.ungrab()
            except OSError:
                pass
        if not on:
            held_move.clear()
            for b in list(held_buttons):
                ui.write(e.EV_KEY, b, 0)
            held_buttons.clear()
            ui.syn()
            hold_start = None
        print("Che do chuot:", "BAT" if on else "TAT")

    try:
        while True:
            timeout = TICK if active else None
            readable, _, _ = select.select(keyboards, [], [], timeout)

            for dev in readable:
                for ev in dev.read():
                    if ev.type != e.EV_KEY:
                        continue
                    code, val = ev.code, ev.value  # 1 = nhan, 0 = nha, 2 = giu

                    if code == TOGGLE_KEY and val == 1:
                        set_active(not active)
                        continue
                    if not active:
                        continue

                    if code in (e.KEY_LEFTSHIFT, e.KEY_RIGHTSHIFT):
                        shift = val != 0
                    elif code in MOVE_KEYS:
                        if val == 1:
                            if not held_move:
                                hold_start = time.monotonic()
                            held_move.add(code)
                        elif val == 0:
                            held_move.discard(code)
                            if not held_move:
                                hold_start = None
                    elif code in BUTTON_KEYS and val in (0, 1):
                        btn = BUTTON_KEYS[code]
                        ui.write(e.EV_KEY, btn, val)
                        ui.syn()
                        (held_buttons.add if val else held_buttons.discard)(btn)
                    elif code in SCROLL_KEYS and val in (1, 2):
                        ui.write(e.EV_REL, e.REL_WHEEL, SCROLL_KEYS[code])
                        ui.syn()

            now = time.monotonic()
            dt = now - last
            last = now

            if active and held_move:
                held_for = now - hold_start
                t = min(held_for / ACCEL_TIME, 1.0)
                speed = BASE_SPEED + (MAX_SPEED - BASE_SPEED) * t
                if shift:
                    speed *= SLOW_FACTOR

                dx = sum(MOVE_KEYS[k][0] for k in held_move)
                dy = sum(MOVE_KEYS[k][1] for k in held_move)
                rem_x += dx * speed * dt
                rem_y += dy * speed * dt
                ix, iy = int(rem_x), int(rem_y)
                rem_x -= ix
                rem_y -= iy
                if ix:
                    ui.write(e.EV_REL, e.REL_X, ix)
                if iy:
                    ui.write(e.EV_REL, e.REL_Y, iy)
                if ix or iy:
                    ui.syn()
            else:
                rem_x = rem_y = 0.0

    except KeyboardInterrupt:
        pass
    finally:
        if active:
            set_active(False)
        ui.close()


if __name__ == "__main__":
    main()
