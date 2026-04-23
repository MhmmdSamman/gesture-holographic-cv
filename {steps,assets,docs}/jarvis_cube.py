import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

W, H = 1280, 720
CX, CY = W // 2, H // 2

angle_x = 30.0
angle_y = 30.0
scale = 200.0
prev_ix, prev_iy = None, None
prev_dist = None

VERTICES = np.array([
    [-1, -1, -1], [ 1, -1, -1],
    [ 1,  1, -1], [-1,  1, -1],
    [-1, -1,  1], [ 1, -1,  1],
    [ 1,  1,  1], [-1,  1,  1],
], dtype=float)

EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

FACES = [
    (0,1,2,3),
    (4,5,6,7),
    (0,1,5,4),
    (2,3,7,6),
    (0,3,7,4),
    (1,2,6,5),
]

FACE_COLORS = [
    (0, 200, 255),
    (0, 150, 255),
    (0, 100, 200),
    (0, 80, 180),
    (0, 120, 220),
    (0, 60, 160),
]


def rotate_x(pts, a):
    c, s = math.cos(a), math.sin(a)
    R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    return pts @ R.T

def rotate_y(pts, a):
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return pts @ R.T

def project(pts, sc):
    result = []
    for p in pts:
        z = p[2] + 3
        if z == 0:
            z = 0.001
        x2d = int(CX + (p[0] * sc) / z)
        y2d = int(CY + (p[1] * sc) / z)
        result.append((x2d, y2d))
    return result

def jari_terangkat(lm, ujung, pangkal):
    return lm[ujung].y < lm[pangkal].y

def hitung_jarak(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def gambar_hud(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "JARVIS HOLOGRAPHIC INTERFACE", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2)
    cv2.putText(frame, f"ROT X:{angle_x:.1f}  ROT Y:{angle_y:.1f}  SCALE:{scale:.0f}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 200), 1)

    cv2.putText(frame, "[ TELUNJUK = ROTATE ]  [ 2 TANGAN = ZOOM ]  [ MENGEPAL = RESET ]",
                (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 180), 1)

    for i in range(0, W, 80):
        cv2.line(frame, (i, 0), (i, H), (0, 30, 40), 1)
    for i in range(0, H, 80):
        cv2.line(frame, (0, i), (W, i), (0, 30, 40), 1)

def gambar_cube(frame, angle_x, angle_y, scale):
    pts = VERTICES.copy()
    pts = rotate_x(pts, math.radians(angle_x))
    pts = rotate_y(pts, math.radians(angle_y))

    projected = project(pts, scale)

    z_rotated = rotate_x(VERTICES.copy(), math.radians(angle_x))
    z_rotated = rotate_y(z_rotated, math.radians(angle_y))

    face_data = []
    for fi, face in enumerate(FACES):
        z_avg = np.mean([z_rotated[v][2] for v in face])
        face_data.append((z_avg, fi, face))

    face_data.sort(key=lambda x: x[0])

    for z_avg, fi, face in face_data:
        pts_face = np.array([projected[v] for v in face], dtype=np.int32)
        overlay = frame.copy()
        color = FACE_COLORS[fi]
        alpha = 0.12 + (z_avg + 2) * 0.04
        alpha = max(0.05, min(0.25, alpha))
        cv2.fillPoly(overlay, [pts_face], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for edge in EDGES:
        p1 = projected[edge[0]]
        p2 = projected[edge[1]]
        z1 = z_rotated[edge[0]][2]
        z2 = z_rotated[edge[1]][2]
        z_avg = (z1 + z2) / 2
        brightness = int(150 + (z_avg + 1.5) * 35)
        brightness = max(80, min(255, brightness))
        color = (0, brightness, 255)
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

    for i, p in enumerate(projected):
        z = z_rotated[i][2]
        bright = int(150 + (z + 1.5) * 40)
        bright = max(100, min(255, bright))
        cv2.circle(frame, p, 5, (0, bright, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 8, (0, bright // 2, 128), 1, cv2.LINE_AA)

    cx_pt = np.mean(projected, axis=0).astype(int)
    cv2.line(frame, (cx_pt[0]-15, cx_pt[1]), (cx_pt[0]+15, cx_pt[1]), (0, 255, 200), 1)
    cv2.line(frame, (cx_pt[0], cx_pt[1]-15), (cx_pt[0], cx_pt[1]+15), (0, 255, 200), 1)


print("JARVIS Interface aktif...")
print("Kontrol:")
print("  Telunjuk → Rotate cube")
print("  2 tangan → Zoom in/out")
print("  Mengepal → Reset posisi")
print("  Q        → Keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = hands.process(rgb)

    gambar_hud(frame)
    gambar_cube(frame, angle_x, angle_y, scale)

    tangan_list = []

    if hasil.multi_hand_landmarks:
        for hand_landmarks in hasil.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 150, 255), thickness=2)
            )
            lm = hand_landmarks.landmark
            ix = int(lm[8].x * W)
            iy = int(lm[8].y * H)
            tx = int(lm[4].x * W)
            ty = int(lm[4].y * H)

            telunjuk_up = jari_terangkat(lm, 8, 6)
            tengah_up   = jari_terangkat(lm, 12, 10)
            manis_up    = jari_terangkat(lm, 16, 14)
            kelingking_up = jari_terangkat(lm, 20, 18)

            mengepal = not telunjuk_up and not tengah_up and not manis_up and not kelingking_up

            tangan_list.append({
                "ix": ix, "iy": iy,
                "tx": tx, "ty": ty,
                "telunjuk_up": telunjuk_up,
                "mengepal": mengepal,
            })

            cv2.circle(frame, (ix, iy), 12, (0, 255, 200), -1)
            cv2.circle(frame, (ix, iy), 18, (0, 200, 255), 2)

        if len(tangan_list) == 1:
            t = tangan_list[0]

            if t["mengepal"]:
                angle_x = 30.0
                angle_y = 30.0
                scale = 200.0
                prev_ix, prev_iy = None, None
                cv2.putText(frame, "RESET", (CX - 40, CY),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 200), 3)

            elif t["telunjuk_up"]:
                if prev_ix is not None and prev_iy is not None:
                    dx = t["ix"] - prev_ix
                    dy = t["iy"] - prev_iy
                    angle_y += dx * 0.4
                    angle_x += dy * 0.4
                prev_ix, prev_iy = t["ix"], t["iy"]

                cv2.line(frame, (CX, CY), (t["ix"], t["iy"]), (0, 100, 180), 1)
                cv2.putText(frame, "ROTATING", (t["ix"] + 15, t["iy"]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
            else:
                prev_ix, prev_iy = None, None

        elif len(tangan_list) == 2:
            p1 = (tangan_list[0]["ix"], tangan_list[0]["iy"])
            p2 = (tangan_list[1]["ix"], tangan_list[1]["iy"])
            dist = hitung_jarak(p1, p2)

            cv2.line(frame, p1, p2, (0, 200, 255), 2)
            mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
            cv2.circle(frame, mid, 8, (0, 255, 200), -1)
            cv2.putText(frame, "ZOOM", (mid[0]+10, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

            if prev_dist is not None:
                delta = dist - prev_dist
                scale += delta * 0.8
                scale = max(80, min(500, scale))
            prev_dist = dist
            prev_ix, prev_iy = None, None
        else:
            prev_dist = None

    cv2.imshow("Holographic Cube", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
