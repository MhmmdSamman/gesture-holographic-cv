import cv2
import mediapipe as mp

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

print("Kamera berjalan... tekan Q untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal baca frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = hands.process(rgb)

    if hasil.multi_hand_landmarks:
        for hand_landmarks in hasil.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cx = int(hand_landmarks.landmark[8].x * frame.shape[1])
            cy = int(hand_landmarks.landmark[8].y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 12, (0, 255, 0), -1)
            cv2.putText(frame, f"Jari: ({cx}, {cy})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tangan tidak terdeteksi", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Step 1 - Deteksi Tangan", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Selesai!")