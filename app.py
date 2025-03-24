import cv2, math
import mediapipe as mp


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def finger_count():
    
    # 0: The camera to use (can be changed)
    cap = cv2.VideoCapture(0)

    # Detection of face, hands and posture
    # Sensitivity between 0 and 1
    interface = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    while cap.isOpened():

        # status: If a frame was captured
        # frame:  The frame captured
        status, frame = cap.read()

        # If no frame detected, finish it
        if not status:
            break

        # VideoCapture gets the frame in BGR format by default
        # So we transform it into RGB
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = interface.process(rgb_frame)

        # Left hand detection
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        # Right hand detection
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
        
        if results.left_hand_landmarks or results.right_hand_landmarks:
            fingers = 0

            if results.left_hand_landmarks:
                fingers += count_fingers_up(results.left_hand_landmarks.landmark)
            
            if results.right_hand_landmarks:
                fingers += count_fingers_up(results.right_hand_landmarks.landmark)
            
            cv2.putText(frame, f"Fingers Up: {fingers}", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Display camera
        cv2.imshow("frame", frame)

        # Close camera if click 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def calculate_distances(a, b):
    x1,y1,z1 = a.x, a.y, a.z
    x2,y2,z2 = b.x, b.y, b.z

    # the distance between the two points
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

def count_fingers_up(hand_landmark):

    # References in finger-div.png
    finger_indices = [8,12,16,20]

    # Wrist location
    wrist_landmark = hand_landmark[0]
    finger_count = 0

    # Check for distance between thumbs and wrist
    thumb_wrist_distance = calculate_distances(hand_landmark[4], wrist_landmark)
    if thumb_wrist_distance > 0.25:
        finger_count += 1
    
    for finger_index in finger_indices:
        finger_landmark = hand_landmark[finger_index]
        finger_wrist_distance = calculate_distances(finger_landmark, wrist_landmark)

        if finger_wrist_distance > 0.3:
            finger_count += 1
    
    return finger_count


finger_count()








