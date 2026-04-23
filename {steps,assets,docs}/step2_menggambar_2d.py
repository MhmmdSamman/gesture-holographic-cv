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
drawing = False
warna = (0, 255, 0)
tebal = 5

def jari_terangkat(landmarks, ujung, pangkal):
    return landmarks[ujung].y < landmarks[pangkal].y

print("Kontrol:")
print("  Telunjuk saja   = MENGGAMBAR")
print("  Telunjuk+Tengah = BERHENTI")
print("  Jempol terangkat = HAPUS CANVAS")
print("  Tekan Q untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = hands.process(rgb)

    h, w, _ = frame.shape
    status = "Buka tangan untuk mulai"

    if hasil.multi_hand_landmarks:
        for hand_landmarks in hasil.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark

            telunjuk_up = jari_terangkat(lm, 8, 6)
            tengah_up   = jari_terangkat(lm, 12, 10)
            jempol_up   = lm[4].x < lm[3].x

            cx = int(lm[8].x * w)
            cy = int(lm[8].y * h)

            if jempol_up and not telunjuk_up and not tengah_up:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                status = "Canvas dihapus!"
                prev_x, prev_y = 0, 0
                drawing = False

            elif telunjuk_up and not tengah_up:
                status = "Menggambar..."
                drawing = True
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = cx, cy
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), warna, tebal)
                prev_x, prev_y = cx, cy

            elif telunjuk_up and tengah_up:
                status = "Berhenti menggambar"
                drawing = False
                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0
                drawing = False

            cv2.circle(frame, (cx, cy), 10, warna, -1)

    gabung = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    cv2.putText(gabung, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(gabung, "Q=Keluar | Jempol=Hapus", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    cv2.imshow("Step 2 - Menggambar 2D", gabung)
    cv2.imshow("Canvas", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
