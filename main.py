import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose class and drawing utility
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Load HOG + SVM based human detector from OpenCV
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Function to detect action based on landmarks
def detect_action(landmarks, image_height, image_width):
    # Get key landmarks (convert to image coordinates)
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    # Convert relative coordinates to absolute image coordinates
    left_shoulder = [int(left_shoulder.x * image_width), int(left_shoulder.y * image_height)]
    right_shoulder = [int(right_shoulder.x * image_width), int(right_shoulder.y * image_height)]
    left_hip = [int(left_hip.x * image_width), int(left_hip.y * image_height)]
    right_hip = [int(right_hip.x * image_width), int(right_hip.y * image_height)]
    left_knee = [int(left_knee.x * image_width), int(left_knee.y * image_height)]
    right_knee = [int(right_knee.x * image_width), int(right_knee.y * image_height)]
    left_ankle = [int(left_ankle.x * image_width), int(left_ankle.y * image_height)]
    right_ankle = [int(right_ankle.x * image_width), int(right_ankle.y * image_height)]

    # Calculate vertical distances between landmarks
    shoulder_hip_dist = np.abs(left_shoulder[1] - left_hip[1])
    knee_ankle_dist = np.abs(left_knee[1] - left_ankle[1])

    # Define thresholds for actions
    min_half_body_dist = 0.3 * image_height  # Half body detection threshold

    # Action classification rules based on distances
    if shoulder_hip_dist > min_half_body_dist and knee_ankle_dist > min_half_body_dist:
        return "Standing"
    elif shoulder_hip_dist > min_half_body_dist and knee_ankle_dist < min_half_body_dist:
        return "Sitting"
    elif shoulder_hip_dist < min_half_body_dist:
        return "Sleeping"
    else:
        return "Unknown"

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect the pose
    results = pose.process(image)
    
    # Convert back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Get the frame dimensions
    h, w, _ = image.shape

    # If pose landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get min and max coordinates for the bounding box
        min_x = int(min([landmark.x for landmark in landmarks]) * w)
        min_y = int(min([landmark.y for landmark in landmarks]) * h)
        max_x = int(max([landmark.x for landmark in landmarks]) * w)
        max_y = int(max([landmark.y for landmark in landmarks]) * h)

        # Draw the bounding box
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        # Detect action
        action = detect_action(landmarks, h, w)

        # Display the detected action in bright green
        cv2.putText(image, f'Action: {action}', (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Human detection using HOG detector
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))

    # Draw bounding boxes around detected humans
    human_count = 0
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        human_count += 1

    # Display human count in bright green
    cv2.putText(image, f'Human Count: {human_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Human Activity Detection', image)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()