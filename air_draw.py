import cv2
import mediapipe as mp
import numpy as np
import datetime
import os

# ── Setup ──────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=1,
                          min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

canvas       = None
prev_x, prev_y = 0, 0
eraser_mode  = False
brush_size   = 8
toast_msg    = ""
toast_timer  = 0

os.makedirs("saved_drawings", exist_ok=True)

# ── Colors ─────────────────────────────────────────
COLORS = [
    ("Red",     (0,   0,   255)),
    ("Orange",  (0,  140,  255)),
    ("Yellow",  (0,  255,  255)),
    ("Green",   (0,  200,    0)),
    ("Cyan",    (255,220,    0)),
    ("Blue",    (255,  0,    0)),
    ("Purple",  (180,  0,  180)),
    ("Pink",    (147, 20,  255)),
    ("White",   (255,255,  255)),
    ("Black",   (  0,  0,    0)),
]
color_index   = 0
current_color = COLORS[color_index][1]

BRUSH_SIZES = [3, 6, 10, 16, 24]
brush_idx   = 1
brush_size  = BRUSH_SIZES[brush_idx]

# ── Finger helpers ─────────────────────────────────
def fingers_up(lm):
    tips  = [8, 12, 16, 20]
    count = sum(1 for t in tips
                if lm.landmark[t].y < lm.landmark[t-2].y)
    thumb = 1 if lm.landmark[4].x < lm.landmark[3].x else 0
    return count, thumb

def is_fist(lm):
    tips = [8, 12, 16, 20]
    return all(lm.landmark[t].y > lm.landmark[t-2].y for t in tips)

def two_fingers(lm):
    idx_up  = lm.landmark[8].y  < lm.landmark[6].y
    mid_up  = lm.landmark[12].y < lm.landmark[10].y
    ring_dn = lm.landmark[16].y > lm.landmark[14].y
    pink_dn = lm.landmark[20].y > lm.landmark[18].y
    return idx_up and mid_up and ring_dn and pink_dn

def show_toast(msg):
    global toast_msg, toast_timer
    toast_msg   = msg
    toast_timer = 60  # frames

# ── UI drawing ─────────────────────────────────────
def draw_ui(frame, h, w):
    # Top bar background
    cv2.rectangle(frame, (0,0), (w, 70), (30,30,30), -1)

    # Color buttons
    for i,(name,col) in enumerate(COLORS):
        x = 10 + i*58
        cv2.rectangle(frame, (x,8), (x+50,42), col, -1)
        if i == color_index and not eraser_mode:
            cv2.rectangle(frame, (x-2,6), (x+52,44), (255,255,255), 2)

    # Brush size buttons
    for i,sz in enumerate(BRUSH_SIZES):
        x = 10 + i*52
        y = 52
        active = (i == brush_idx)
        cv2.rectangle(frame, (x,y), (x+44,y+14),
                      (80,80,80) if not active else (200,200,200), -1)
        cv2.putText(frame, f"B{sz}", (x+4, y+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0,0,0) if active else (200,200,200), 1)

    # Right buttons
    # ERASE
    ecol = (0,200,200) if eraser_mode else (60,60,60)
    cv2.rectangle(frame, (w-270,8), (w-190,42), ecol, -1)
    cv2.putText(frame, "ERASE", (w-265,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    # CLEAR
    cv2.rectangle(frame, (w-180,8), (w-100,42), (60,60,60), -1)
    cv2.putText(frame, "CLEAR", (w-175,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    # SAVE
    cv2.rectangle(frame, (w-90,8), (w-10,42), (0,160,0), -1)
    cv2.putText(frame, "SAVE", (w-85,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

# ── Main loop ──────────────────────────────────────
print("✅ Air Canvas started!")
print("Gestures: 1 finger=Draw | 2 fingers=Next color")
print("          Fist=Clear   | Open palm=Stop drawing")
print("Keyboard: S=Save C=Clear E=Erase Q=Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_ui(frame, h, w)

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            ix = int(hand_lm.landmark[8].x * w)
            iy = int(hand_lm.landmark[8].y * h)
            fc, thumb = fingers_up(hand_lm)

            # ── Gesture shortcuts ──────────────────
            # Fist = clear canvas
            if is_fist(hand_lm):
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                prev_x, prev_y = 0, 0
                show_toast("🗑 Canvas Cleared!")

            # 2 fingers = next color
            elif two_fingers(hand_lm) and iy > 70:
                color_index   = (color_index + 1) % len(COLORS)
                current_color = COLORS[color_index][1]
                eraser_mode   = False
                prev_x, prev_y = 0, 0
                show_toast(f"🎨 Color: {COLORS[color_index][0]}")

            # 1 finger = draw
            elif fc == 1 and iy > 70:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy
                draw_col = (0,0,0) if eraser_mode else current_color
                sz       = 30     if eraser_mode else brush_size
                cv2.line(canvas, (prev_x,prev_y),
                         (ix,iy), draw_col, sz)
                prev_x, prev_y = ix, iy
                cv2.circle(frame, (ix,iy), sz//2,
                           draw_col if not eraser_mode
                           else (100,100,100), -1)

            # Pointing at UI buttons (finger in top bar)
            elif iy < 70 and fc >= 1:
                # Color buttons
                for i,(name,col) in enumerate(COLORS):
                    x = 10 + i*58
                    if x < ix < x+50 and 8 < iy < 42:
                        color_index   = i
                        current_color = col
                        eraser_mode   = False
                        show_toast(f"🎨 {name}")
                # Brush size
                for i,sz in enumerate(BRUSH_SIZES):
                    x = 10 + i*52
                    if x < ix < x+44 and 52 < iy < 66:
                        brush_idx  = i
                        brush_size = BRUSH_SIZES[i]
                        show_toast(f"✏️ Brush: {sz}")
                # ERASE button
                if w-270 < ix < w-190 and 8 < iy < 42:
                    eraser_mode = not eraser_mode
                    show_toast("🧹 Eraser ON" if eraser_mode
                               else "✏️ Draw mode")
                # CLEAR button
                if w-180 < ix < w-100 and 8 < iy < 42:
                    canvas = np.zeros((h,w,3), dtype=np.uint8)
                    show_toast("🗑 Cleared!")
                # SAVE button
                if w-90 < ix < w-10 and 8 < iy < 42:
                    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = f"saved_drawings/drawing_{ts}.png"
                    cv2.imwrite(path, canvas)
                    show_toast(f"💾 Saved!")

                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0
    else:
        prev_x, prev_y = 0, 0

    # Merge canvas onto frame
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask     = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    mask_inv    = cv2.bitwise_not(mask)
    frame_bg    = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg   = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined    = cv2.add(frame_bg, canvas_fg)

    draw_ui(combined, h, w)

    # Status bar
    mode_txt = "ERASER" if eraser_mode else \
               f"DRAW  Color:{COLORS[color_index][0]}  Brush:{brush_size}"
    cv2.putText(combined, mode_txt, (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    # Gesture hints
    cv2.putText(combined,
                "1finger=Draw | 2finger=Color | Fist=Clear | Palm=Stop",
                (10, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140,140,140), 1)

    # Toast notification
    if toast_timer > 0:
        cv2.rectangle(combined, (w//2-160, h//2-30),
                      (w//2+160, h//2+10), (30,30,30), -1)
        cv2.putText(combined, toast_msg, (w//2-150, h//2+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,180), 2)
        toast_timer -= 1

    cv2.imshow("Air Canvas", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((h,w,3), dtype=np.uint8)
        show_toast("🗑 Cleared!")
    elif key == ord('e'):
        eraser_mode = not eraser_mode
        show_toast("🧹 Eraser ON" if eraser_mode else "✏️ Draw mode")
    elif key == ord('s'):
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"saved_drawings/drawing_{ts}.png"
        cv2.imwrite(path, canvas)
        show_toast(f"💾 Saved to saved_drawings/")

cap.release()
cv2.destroyAllWindows()