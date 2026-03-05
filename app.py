import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Get camera size
ret, frame = cap.read()
frame = cv2.flip(frame, 1)
H, W = frame.shape[:2]

canvas = np.zeros((H, W, 3), dtype=np.uint8)

BLOCK = 40
blocks = set()

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def draw_grid(img):
    for x in range(0, W, BLOCK):
        cv2.line(img, (x, 0), (x, H), (40, 40, 40), 1)
    for y in range(0, H, BLOCK):
        cv2.line(img, (0, y), (W, y), (40, 40, 40), 1)

def rounded_rect(img, p1, p2, color):
    x1, y1 = p1
    x2, y2 = p2
    r = 8
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
    cv2.circle(img, (x1+r, y1+r), r, color, -1)
    cv2.circle(img, (x2-r, y1+r), r, color, -1)
    cv2.circle(img, (x1+r, y2-r), r, color, -1)
    cv2.circle(img, (x2-r, y2-r), r, color, -1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    left, right = None, None
    lp, rp = False, False

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand, info in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = info.classification[0].label
            i = hand.landmark[8]
            t = hand.landmark[4]

            ix, iy = int(i.x * W), int(i.y * H)
            tx, ty = int(t.x * W), int(t.y * H)
            pinch = dist((ix, iy), (tx, ty)) < 30

            if label == "Left":
                left, lp = (ix, iy), pinch
            else:
                right, rp = (ix, iy), pinch

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Logic
    if left and not lp:
        bx = (left[0] // BLOCK) * BLOCK
        by = (left[1] // BLOCK) * BLOCK
        blocks.add((bx, by))

    if lp and rp and left:
        bx = (left[0] // BLOCK) * BLOCK
        by = (left[1] // BLOCK) * BLOCK
        blocks.discard((bx, by))

    # Canvas
    canvas[:] = (15, 15, 15)
    draw_grid(canvas)

    # Preview block
    if left:
        px = (left[0] // BLOCK) * BLOCK
        py = (left[1] // BLOCK) * BLOCK
        cv2.rectangle(canvas, (px, py), (px+BLOCK, py+BLOCK), (80, 80, 80), 2)

    # Draw blocks
    for x, y in blocks:
        rounded_rect(canvas, (x+2, y+2), (x+BLOCK-2, y+BLOCK-2), (0, 180, 255))

    # UI panel
    cv2.rectangle(canvas, (0, 0), (W, 50), (20, 20, 20), -1)
    cv2.putText(canvas,
                "Left hand: Draw   |   Both pinch: Erase   |   ESC: Exit",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (200, 200, 200),
                2)

    final = cv2.addWeighted(frame, 0.55, canvas, 0.9, 0)
    cv2.imshow("Gesture Block Writer", final)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()