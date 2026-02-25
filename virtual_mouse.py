"""
Virtual Mouse - Fixed Edition
pip install opencv-python mediapipe pyautogui numpy pycaw comtypes

Gestures:
  ☝️  Index only          = Move cursor
  ✌️  Index + Middle      = Single click (with cooldown, no repeat)
  👌  Thumb+Index close   = Volume Down
  🖐️  Thumb+Index wide    = Volume Up

Press Q to quit.
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import collections

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

# ── Volume ────────────────────────────────────────────────────────────────────
HAS_VOL = False
_vol    = None
VMIN    = -65.0
VMAX    =   0.0

try:
    import comtypes
    comtypes.CoInitialize()
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    _devices = AudioUtilities.GetSpeakers()
    _iface   = _devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    _vol     = cast(_iface, POINTER(IAudioEndpointVolume))
    VMIN, VMAX = _vol.GetVolumeRange()[:2]
    HAS_VOL  = True
    print("[OK] Volume control ready")
except Exception as e:
    print(f"[INFO] pycaw not available — using keyboard keys  ({e})")

def get_vol():
    if HAS_VOL:
        return (_vol.GetMasterVolumeLevel() - VMIN) / (VMAX - VMIN)
    return 0.5

def vol_up():
    if HAS_VOL:
        _vol.SetMasterVolumeLevel(
            min(VMAX, _vol.GetMasterVolumeLevel() + (VMAX - VMIN) * 0.03), None)
    else:
        pyautogui.press("volumeup")

def vol_down():
    if HAS_VOL:
        _vol.SetMasterVolumeLevel(
            max(VMIN, _vol.GetMasterVolumeLevel() - (VMAX - VMIN) * 0.03), None)
    else:
        pyautogui.press("volumedown")

# ── Settings ──────────────────────────────────────────────────────────────────
SW, SH   = pyautogui.size()
CW, CH   = 640, 480
MARGIN   = 0.15
X1, X2   = int(CW * MARGIN), int(CW * (1 - MARGIN))
Y1, Y2   = int(CH * MARGIN), int(CH * (1 - MARGIN))
SMOOTH   = 5
VOL_WIDE = 130
VOL_CLSE = 65
VOL_GAP  = 0.04
CLICK_COOLDOWN = 0.8    # seconds before another click is allowed

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode        = False,
    max_num_hands            = 1,
    model_complexity         = 0,
    min_detection_confidence = 0.7,
    min_tracking_confidence  = 0.6,
)

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CH)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    exit()

print(f"Screen {SW}x{SH}  |  Camera {CW}x{CH}")
print("☝ move  ✌ click  👌 vol dn  🖐 vol up  |  Q = quit")

# ── State ─────────────────────────────────────────────────────────────────────
xs         = collections.deque(maxlen=SMOOTH)
ys         = collections.deque(maxlen=SMOOTH)
prev_t     = time.time()
vol_t      = 0.0
click_t    = 0.0          # last time a click fired
click_held = False        # True while ✌ gesture is active (to fire only once)
cur_vol    = get_vol()    # cached volume so bar updates immediately
FONT       = cv2.FONT_HERSHEY_SIMPLEX

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]

    now    = time.time()
    fps    = 1.0 / max(now - prev_t, 1e-9)
    prev_t = now

    rgb               = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res               = hands.process(rgb)
    rgb.flags.writeable = True

    gesture = "-"

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark

        mp_draw.draw_landmarks(frame,
            res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        # Finger states
        thumb  = lm[4].x  < lm[3].x
        index  = lm[8].y  < lm[6].y
        middle = lm[12].y < lm[10].y
        ring   = lm[16].y < lm[14].y
        pinky  = lm[20].y < lm[18].y

        ix = int(lm[8].x * W);  iy = int(lm[8].y * H)
        tx = int(lm[4].x * W);  ty = int(lm[4].y * H)
        td = math.hypot(ix - tx, iy - ty)

        # ── ✌️ CLICK — fires ONCE per gesture, resets when hand opens ─────────
        if index and middle and not thumb and not ring and not pinky:
            gesture = "CLICK"
            cv2.circle(frame, (ix, iy), 16, (0, 0, 255), -1)
            # Only fire click the first frame the gesture appears
            if not click_held and (now - click_t) > CLICK_COOLDOWN:
                pyautogui.click()
                click_t    = now
                click_held = True
                print(f"[click] at {pyautogui.position()}")
        else:
            # Reset so next ✌ gesture can fire again
            click_held = False

        # ── ☝️ MOVE ──────────────────────────────────────────────────────────
        if index and not middle and not thumb and not ring and not pinky:
            cx = max(X1, min(X2, ix))
            cy = max(Y1, min(Y2, iy))
            xs.append(np.interp(cx, [X1, X2], [0, SW]))
            ys.append(np.interp(cy, [Y1, Y2], [0, SH]))
            pyautogui.moveTo(int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
            gesture = "MOVE"
            cv2.circle(frame, (ix, iy), 12, (0, 255, 0), -1)

        # ── 🖐️👌 VOLUME ──────────────────────────────────────────────────────
        elif thumb and index and not middle and not ring and not pinky:
            cv2.line(frame, (tx, ty), (ix, iy), (0, 210, 100), 2)
            cv2.putText(frame, f"{int(td)}px",
                        (int((tx+ix)/2)+5, int((ty+iy)/2)),
                        FONT, 0.45, (0, 210, 100), 1)
            if td > VOL_WIDE and now > vol_t:
                cur_vol = min(1.0, cur_vol + 0.03)   # update locally first
                vol_up()
                vol_t   = now + VOL_GAP
                gesture = f"VOL UP {int(cur_vol*100)}%"
            elif td < VOL_CLSE and now > vol_t:
                cur_vol = max(0.0, cur_vol - 0.03)   # update locally first
                vol_down()
                vol_t   = now + VOL_GAP
                gesture = f"VOL DN {int(cur_vol*100)}%"
            else:
                gesture = f"VOL  {int(cur_vol*100)}%"

    # ── Active zone ───────────────────────────────────────────────────────────
    cv2.rectangle(frame, (X1, Y1), (X2, Y2), (0, 200, 255), 1)

    # ── Volume bar — uses cached cur_vol so it updates instantly ──────────────
    bh = int(160 * cur_vol)
    cv2.rectangle(frame, (W-36, 60),     (W-10, 220),     (40, 40, 40),  -1)
    cv2.rectangle(frame, (W-36, 220-bh), (W-10, 220),     (0, 210, 100), -1)
    cv2.putText(frame, f"{int(cur_vol*100)}%", (W-42, 236), FONT, 0.4, (0,210,100), 1)
    cv2.putText(frame, "VOL",                  (W-34,  55), FONT, 0.38, (140,140,140), 1)

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (W, 50), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS:{fps:.0f}  {gesture}",
                (10, 34), FONT, 0.8, (0, 230, 120), 2)

    cv2.imshow("Virtual Mouse  [Q = quit]", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
hands.close()
cv2.destroyAllWindows()
print("Bye!")