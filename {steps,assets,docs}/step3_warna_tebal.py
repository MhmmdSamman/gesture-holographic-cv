import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0
warna_aktif = (0, 255, 0)
tebal = 5

WARNA = [
    {"nama": "Hijau",  "bgr": (0, 255, 0)},
    {"nama": "Merah",  "bgr": (0, 0, 255)},
    {"nama": "Biru",   "bgr": (255, 0, 0)},
    {"nama": "Kuning", "bgr": (0, 255, 255)},
    {"nama": "Pink",   "bgr": (255, 0, 255)},
    {"nama": "Putih",  "bgr": (255, 255, 255)},
    {"nama": "Hapus",  "bgr": (0, 0, 0)},
]

TEBAL_LIST = [3, 6, 10, 16]
tebal_idx = 1


def jari_terangkat(lm, ujung, pangkal):
    return lm[ujung].y < lm[pangkal].y


def gambar_toolbar(frame):
    bar_h = 60
    cv2.rectangle(frame, (0, 0), (640, bar_h), (25, 25, 25), -1)

    kotak_w = 60
    for i, w in enumerate(WARNA):
        x1 = i * kotak_w + 5
        x2 = x1 + kotak_w - 10
        cv2.rectangle(frame, (x1, 8), (x2, 52), w["bgr"], -1)
        if w["bgr"] == warna_aktif:
            cv2.rectangle(frame, (x1 - 2, 6), (x2 + 2, 54), (255, 255, 255), 2)
        if w["nama"] == "Hapus":
            cv2.putText(frame, "X", (x1 + 15, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for i, t in enumerate(TEBAL_LIST):
        x = 450 + i * 48
        y = 30
        cv2.circle(frame, (x, y), t + 2, (200, 200, 200), -1)
        if i == tebal_idx:
            cv2.circle(frame, (x, y), t + 5, (255, 255, 255), 2)

    cv2.putText(frame, "Warna:", (5, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(frame, "Tebal:", (450, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def cek_toolbar_klik(cx, cy):
    global warna_aktif, tebal, tebal_idx

    if cy < 60:
        kotak_w = 60
        for i, w in enumerate(WARNA):
            x1 = i * kotak_w + 5
            x2 = x1 + kotak_w - 10
            if x1 < cx < x2:
                if w["nama"] == "Hapus":
                    warna_aktif = (0, 0, 0)
                else:
                    warna_aktif = w["bgr"]
                return True

        for i, t in enumerate(TEBAL_LIST):
            x = 450 + i * 48
            if abs(cx - x) < 20:
                tebal_idx = i
                tebal = TEBAL_LIST[i]
                return True

    return False


print("Kontrol:")
print("  Telunjuk saja        = MENGGAMBAR")
print("  Telunjuk + Tengah    = BERHENTI")
print("  Jempol terangkat     = HAPUS CANVAS")
print("  Arahkan jari ke toolbar atas untuk ganti warna & tebal")
print("  Tekan Q untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = hands.process(rgb)

    h, w, _ = frame.shape
    status = "Tangan tidak terdeteksi"

    if hasil.multi_hand_landmarks:
        for hand_landmarks in hasil.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark

            telunjuk_up = jari_terangkat(lm, 8, 6)
            tengah_up   = jari_terangkat(lm, 12, 10)
            jempol_up   = lm[4].x < lm[3].x

            cx = int(lm[8].x * w)
            cy = int(lm[8].y * h)

            di_toolbar = cek_toolbar_klik(cx, cy)

            if jempol_up and not telunjuk_up and not tengah_up:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                status = "Canvas dihapus!"
                prev_x, prev_y = 0, 0

            elif di_toolbar:
                status = f"Pilih warna/tebal di toolbar"
                prev_x, prev_y = 0, 0

            elif telunjuk_up and not tengah_up:
                status = "Menggambar..."
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = cx, cy
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), warna_aktif, tebal)
                prev_x, prev_y = cx, cy

            elif telunjuk_up and tengah_up:
                status = "Berhenti menggambar"
                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

            cv2.circle(frame, (cx, cy), 10, warna_aktif, -1)

    gabung = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    gambar_toolbar(gabung)

    cv2.putText(gabung, status, (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(gabung, "Q=Keluar | Jempol=Hapus Canvas", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    cv2.imshow("Step 3 - Warna & Tebal", gabung)
    cv2.imshow("Canvas", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
