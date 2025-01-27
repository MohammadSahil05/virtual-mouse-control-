import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Variables to store previous state
last_click_time = time.time()
click_delay = 0.2  # Prevent double clicks too fast
is_dragging = False
sensitivity = 2  # Adjust this value for faster/slower response

# Function to detect gestures and perform actions
def detect_gestures(hand_landmarks):
    global is_dragging, last_click_time

    # Get necessary landmarks
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Calculate distance between index finger and thumb
    distance = ((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2) ** 0.5
    pinch_threshold = 0.05  # Adjust based on your camera settings

    # Left Click with Pinch Gesture
    if distance < pinch_threshold and (time.time() - last_click_time) > click_delay:
        pyautogui.click()
        last_click_time = time.time()

    # Drag and Drop Gesture
    if distance < pinch_threshold:
        if not is_dragging:
            pyautogui.mouseDown()  # Start dragging
            is_dragging = True
    else:
        if is_dragging:
            pyautogui.mouseUp()  # Stop dragging
            is_dragging = False

    # Right Click Gesture
    if pinky_tip.y < index_finger_tip.y:
        pyautogui.rightClick()  # Right click

    # Swipe Left/Right for Navigation
    if index_finger_tip.x < thumb_tip.x - pinch_threshold:
        pyautogui.hotkey('alt', 'left')  # Go back
    elif index_finger_tip.x > thumb_tip.x + pinch_threshold:
        pyautogui.hotkey('alt', 'right')  # Go forward

    # Swipe Up/Down for Scrolling
    if middle_finger_tip.y < index_finger_tip.y - pinch_threshold:
        pyautogui.scroll(10)  # Scroll up
    elif middle_finger_tip.y > index_finger_tip.y + pinch_threshold:
        pyautogui.scroll(-10)  # Scroll down

    # Pinch Gesture for Zooming
    if distance < pinch_threshold:
        pyautogui.hotkey('ctrl', '+')  # Zoom in
    else:
        pyautogui.hotkey('ctrl', '-')  # Zoom out

    # Fist Gesture to Minimize
    if thumb_tip.y < index_finger_tip.y and middle_finger_tip.y < index_finger_tip.y:
        pyautogui.hotkey('win', 'd')  # Show desktop

    # Thumbs Up Gesture to Open Task Manager
    if thumb_tip.y < index_finger_tip.y and pinky_tip.y < index_finger_tip.y:
        pyautogui.hotkey( 'esc')  # Open Task Manager

# Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger_tip.x * w)
            y = int(index_finger_tip.y * h)

            # Move the mouse to the index finger's position
            if is_dragging:
                pyautogui.moveTo(x, y, duration=0)
            else:
                pyautogui.moveTo(int(x * sensitivity), int(y * sensitivity))

            # Detect gestures
            detect_gestures(hand_landmarks)

            # Draw landmarks on the frame for visualization (optional)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
