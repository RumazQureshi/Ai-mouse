import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Smoothing variables
prev_x, prev_y = 0, 0
smooth_factor = 5  # Adjust this for smoother movement

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip for natural movement
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            index_x, index_y = 0, 0  # Default values
            thumb_x, thumb_y = 0, 0  # Default values

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index Finger
                    index_x = np.interp(x, [0, frame_width], [0, screen_width])
                    index_y = np.interp(y, [0, frame_height], [0, screen_height])
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

                if id == 4:  # Thumb
                    thumb_x = np.interp(x, [0, frame_width], [0, screen_width])
                    thumb_y = np.interp(y, [0, frame_height], [0, screen_height])
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

            if index_x and index_y:
                # Smooth cursor movement using exponential moving average
                smoothed_x = (prev_x * (smooth_factor - 1) + index_x) / smooth_factor
                smoothed_y = (prev_y * (smooth_factor - 1) + index_y) / smooth_factor

                pyautogui.moveTo(smoothed_x, smoothed_y, duration=0.1)
                prev_x, prev_y = smoothed_x, smoothed_y

            # Click if the index and thumb are close
            if thumb_x and thumb_y and index_x and index_y:
                distance = np.hypot(thumb_x - index_x, thumb_y - index_y)
                if distance < 30:  # Click when fingers are close
                    pyautogui.click()
                    pyautogui.sleep(0.5)

    cv2.imshow("AI Mouse", frame)
    if cv2.waitKey(10) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()