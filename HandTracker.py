import cv2
import mediapipe as mp
import numpy as np
import pyfirmata

# Set video capture dimensions
ws, hs = 1280, 720

# Arduino setup
port = "COM6"
board = pyfirmata.Arduino(port)
servo_pinX = board.get_pin('d:10:s')  # X-axis servo
servo_pinY = board.get_pin('d:11:s')  # Y-axis servo

# Video capture setup
cap = cv2.VideoCapture(0)
cap.set(3, ws)
cap.set(4, hs)

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.6):
        self.hands = mp.solutions.hands.Hands(static_image_mode, max_num_hands, model_complexity, min_detection_confidence, min_tracking_confidence)
        self.drawer = mp.solutions.drawing_utils

    def find_hand(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        process = self.hands.process(rgb_img)
        Hands = []
        h, w, _ = img.shape
        if process.multi_hand_landmarks:
            for hand_lms in process.multi_hand_landmarks:
                lm = hand_lms.landmark[8]  # Index fingertip
                x, y, z = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                Hands.append({"hlm_list": [[x, y, z]]})
                self.drawer.draw_landmarks(img, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)
        return Hands, img

detector = HandDetector()

while True:
    _, img = cap.read()
    hands, img = detector.find_hand(img)
    if hands:
        f1, f2 = hands[0]["hlm_list"][0][:2]
        cv2.circle(img, (f1, f2), 10, (43, 237, 5), -1)
        cv2.line(img, (0, f2), (ws, f2), (33, 8, 110), 1)
        cv2.line(img, (f1, hs), (f1, 0), (33, 8, 110), 1)
        x = np.interp(f1, [0, ws], [180, 0])
        y = np.interp(f2, [0, hs], [180, 0])
        if 0 < x < 180 and 0 < y < 180:
            servo_pinX.write(x)
            servo_pinY.write(y)
    cv2.imshow("Turret", cv2.flip(img, 1))
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        break
