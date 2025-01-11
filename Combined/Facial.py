import os
import cv2
import time
from datetime import datetime
from deepface import DeepFace

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directories
known_faces_dir = '/home/wxn/PycharmProjects/MENGProject/DeepFace/Known_Faces'
output_base_directory = '/home/wxn/PycharmProjects/MENGProject/output/'
processed_base_directory = '/home/wxn/PycharmProjects/MENGProject/processed/'

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

# Create a new output directory for storing processed videos
def create_output_directory(base_path):
    current_time = datetime.now().strftime("%Y_%m_%d_%H:%M")
    i = 1
    dir_name = f"{current_time}_observation_{i}"
    while os.path.exists(os.path.join(base_path, dir_name)):
        i += 1
        dir_name = f"{current_time}_observation_{i}"
    output_path = os.path.join(base_path, dir_name)
    os.makedirs(output_path)
    return output_path

# Function to process a video and save the processed video to the output directory
def process_video(input_video_path, output_directory):
    cap = cv2.VideoCapture(input_video_path)

    # Define codec and create VideoWriter object to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_video_path = os.path.join(output_directory, os.path.basename(input_video_path))

    print(f"Processing video: {input_video_path}")
    print(f"Saving processed video to: {output_video_path}")

    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            if w > 0 and h > 0:
                padding = 10
                x = max(x - padding, 0)
                y = max(y - padding, 0)
                w += 2 * padding
                h += 2 * padding
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_frame = frame[y:y + h, x:x + w]

                try:
                    analysis = DeepFace.analyze(face_frame, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
                    result = analysis[0]

                    # Extract information
                    avg_age = result['age']
                    avg_gender = result['dominant_gender']
                    avg_emotion = result['dominant_emotion']
                    avg_race = result['dominant_race']

                    known_person = is_known_face(face_frame)
                    label = f"Name: {known_person or 'Unknown'}, Age: {avg_age:.1f}, Gender: {avg_gender}, Emotion: {avg_emotion}, Race: {avg_race}"

                    # Add label to the video frame
                    y_label = y - 10 if y - 10 > 10 else y + h + 20
                    for i, line in enumerate(label.split(', ')):
                        cv2.putText(frame, line, (x, y_label + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

                except Exception as e:
                    print(f"Error during analysis: {e}")

        out.write(frame)

    cap.release()
    out.release()
    print(f"Video processing complete: {output_video_path}")

# Function to continuously process new videos in the current directory
def process_videos_in_current_directory(latest_directory, output_directory):
    processed_files = set()

    while True:
        video_files = sorted([f for f in os.listdir(latest_directory) if f.endswith('.mp4') or f.endswith('.avi')])
        new_files = [f for f in video_files if f not in processed_files]

        if not new_files:
            print("No new videos to process. Waiting for new videos...")
            time.sleep(5)  # Wait for 5 seconds before checking for new files again
            continue

        for video_file in new_files:
            video_path = os.path.join(latest_directory, video_file)
            process_video(video_path, output_directory)
            processed_files.add(video_file)

        print("Waiting for new videos to be added to the current directory...")
        time.sleep(5)

# Main function to start monitoring the current directory
def monitor_output_folder():
    # Get the current time directory (e.g., created at 2:43) in the output folder
    now = datetime.now().strftime("%Y_%m_%d_%H:%M")
    directory_name = f"{now}_processed_1"  # Assuming the directory will be created with this name format
    latest_directory = os.path.join(output_base_directory, directory_name)

    while not os.path.exists(latest_directory):
        print(f"Directory {latest_directory} not found. Retrying in 5 seconds...")
        time.sleep(5)

    # Create a new directory in the processed folder
    output_directory = create_output_directory(processed_base_directory)
    print(f"Monitoring folder: {latest_directory}")
    print(f"Processed videos will be saved in: {output_directory}")

    # Continuously monitor and process videos inside the current directory
    process_videos_in_current_directory(latest_directory, output_directory)

if __name__ == "__main__":
    load_known_faces()
    monitor_output_folder()
