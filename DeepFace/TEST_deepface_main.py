import cv2
from deepface import DeepFace
import numpy as np
import os
import time
import threading
from queue import Queue

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory containing images of known individuals
known_faces_dir = '/home/wxn/PycharmProjects/MENGProject/DeepFace/Known_Faces'

# Load known faces
known_faces = []
known_names = []

def load_known_faces():
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(known_faces_dir, filename)
            img = cv2.imread(img_path)
            known_faces.append(img)
            known_names.append(os.path.splitext(filename)[0])

def is_known_face(face_frame):
    for known_face, name in zip(known_faces, known_names):
        try:
            result = DeepFace.verify(face_frame, known_face, enforce_detection=False)
            if result['verified']:
                return name
        except Exception as e:
            print(f"Error during verification: {e}")
    return None

def display_info_window(face_image, label, window_name):
    info_frame = np.zeros((300, 400, 3), dtype=np.uint8)
    face_image_resized = cv2.resize(face_image, (200, 200))
    info_frame[:200, :200] = face_image_resized
    for i, line in enumerate(label.split(', ')):
        cv2.putText(info_frame, line, (210, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(window_name, info_frame)

def analyze_faces(face_queue, result_queue):
    while True:
        face_id, face_frame = face_queue.get()
        try:
            # Analyze the face
            analysis = DeepFace.analyze(
                face_frame,
                actions=['age', 'gender', 'emotion', 'race'],
                enforce_detection=True
            )
            result_queue.put((face_id, analysis))
        except Exception as e:
            print(f"Error during analysis: {e}")
            result_queue.put((face_id, None))
        face_queue.task_done()

def analyze_webcam():
    device_indices = [0, 1]  # List of device indices to try
    cap = None

    for device_index in device_indices:
        print(f"Trying device index: {device_index}")
        cap = cv2.VideoCapture(device_index)

        if cap.isOpened():
            print(f"Successfully opened device index: {device_index}")
            break
        else:
            print(f"Could not open device index: {device_index}")
            cap.release()
    else:
        print("Error: Could not open any video device.")
        return

    load_known_faces()  # Load the known faces from the directory

    face_data = {}
    max_frames = 10  # Number of frames to average over
    time_limit = 300  # 5 minutes in seconds

    face_queue = Queue()
    result_queue = Queue()

    # Start analysis thread
    analysis_thread = threading.Thread(target=analyze_faces, args=(face_queue, result_queue))
    analysis_thread.setDaemon(True)
    analysis_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        current_time = time.time()

        # Draw bounding boxes around detected faces and enqueue them for analysis
        for idx, (x, y, w, h) in enumerate(faces):
            if w > 0 and h > 0:  # Ensure positive width and height
                # Increase the bounding box size slightly
                padding = 10
                x = max(x - padding, 0)
                y = max(y - padding, 0)
                w += 2 * padding
                h += 2 * padding
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_id = f"face_{idx}"

                # Initialize or update data for this face
                if face_id not in face_data:
                    face_data[face_id] = {
                        'age_values': [],
                        'gender_values': [],
                        'emotion_values': [],
                        'race_values': [],
                        'frame_count': 0,
                        'locked_label': None,
                        'label_color': (255, 0, 0),  # Default to blue
                        'last_seen': current_time
                    }
                else:
                    face_data[face_id]['last_seen'] = current_time  # Update last seen time

                face_frame = frame[y:y + h, x:x + w]

                # Enqueue face for analysis
                face_queue.put((face_id, face_frame))

                # Draw the label outside the bounding box if locked
                if face_data[face_id]['locked_label']:
                    y_label = y - 10 if y - 10 > 10 else y + h + 20
                    for i, line in enumerate(face_data[face_id]['locked_label'].split(', ')):
                        cv2.putText(frame, line, (x, y_label + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_data[face_id]['label_color'], 2, cv2.LINE_AA)

        # Process results from the analysis thread
        while not result_queue.empty():
            face_id, analysis = result_queue.get()
            if analysis:
                result = analysis[0]
                face_data[face_id]['age_values'].append(result['age'])
                face_data[face_id]['gender_values'].append(result['dominant_gender'])
                face_data[face_id]['emotion_values'].append(result['dominant_emotion'])
                face_data[face_id]['race_values'].append(result['dominant_race'])

                face_data[face_id]['frame_count'] += 1

                if face_data[face_id]['frame_count'] >= max_frames:
                    # Calculate average values
                    avg_age = sum(face_data[face_id]['age_values']) / len(face_data[face_id]['age_values'])
                    avg_gender = max(set(face_data[face_id]['gender_values']), key=face_data[face_id]['gender_values'].count)
                    avg_emotion = max(set(face_data[face_id]['emotion_values']), key=face_data[face_id]['emotion_values'].count)
                    avg_race = max(set(face_data[face_id]['race_values']), key=face_data[face_id]['race_values'].count)

                    # Check if the person is known
                    known_person = is_known_face(face_frame)
                    if known_person:
                        face_data[face_id]['locked_label'] = f"Name: {known_person}, Age: {avg_age:.1f}, Gender: {avg_gender}, Emotion: {avg_emotion}, Race: {avg_race}"
                        face_data[face_id]['label_color'] = (255, 0, 0)  # Blue
                    else:
                        face_data[face_id]['locked_label'] = f"Unknown, Age: {avg_age:.1f}, Gender: {avg_gender}, Emotion: {avg_emotion}, Race: {avg_race}"
                        face_data[face_id]['label_color'] = (0, 0, 255)  # Red

                    print(f"Locked Label for {face_id}: {face_data[face_id]['locked_label']}")  # Debugging line to ensure label is created

                    # Display the information in a new window
                    display_info_window(face_frame, face_data[face_id]['locked_label'], face_id)

                    # Reset lists and frame count
                    face_data[face_id]['age_values'].clear()
                    face_data[face_id]['gender_values'].clear()
                    face_data[face_id]['emotion_values'].clear()
                    face_data[face_id]['race_values'].clear()
                    face_data[face_id]['frame_count'] = 0

        # Close windows for faces not seen for more than 5 minutes
        for face_id in list(face_data.keys()):
            if current_time - face_data[face_id]['last_seen'] > time_limit:
                cv2.destroyWindow(face_id)
                del face_data[face_id]
                print(f"Closed window for {face_id} due to inactivity.")

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

analyze_webcam()
