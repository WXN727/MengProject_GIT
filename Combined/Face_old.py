import os
import cv2
import time
import threading
import tkinter as tk
from queue import Queue
from datetime import datetime
from deepface import DeepFace
import mediapipe as mp

# MediaPipe Face Detection setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

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

def analyze_faces(face_queue, result_queue):
    while True:
        face_id, face_frame = face_queue.get()
        try:
            # Analyze the face
            analysis = DeepFace.analyze(face_frame, actions=[
                #'age', 'gender',
                'emotion'
                #, 'race'
            ], enforce_detection=False)
            result_queue.put((face_id, analysis))
        except Exception as e:
            print(f"Error during analysis: {e}")
            result_queue.put((face_id, None))
        face_queue.task_done()

class VideoPlayer:
    def __init__(self, base_directory, window_width=1280, window_height=960):
        self.base_directory = base_directory
        self.window_width = window_width
        self.window_height = window_height
        self.directory = self.get_latest_directory()
        self.video_files = self.get_video_files()
        self.current_index = 0
        self.stop = False

        load_known_faces()  # Load known faces

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Video Player")

        # Create a label to display the video
        self.label = tk.Label(self.root)
        self.label.pack()

        # Create a label to display the analysis result
        self.analysis_label = tk.Label(self.root, text="")
        self.analysis_label.pack()

        # Start the video playback in a separate thread
        self.video_thread = threading.Thread(target=self.play_videos)
        self.video_thread.start()

        # Start the GUI main loop
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def get_latest_directory(self):
        while True:
            now = datetime.now()
            date_str = now.strftime("%Y_%m_%d_%H:%M")
            directories = [d for d in os.listdir(self.base_directory) if d.startswith(date_str)]
            if directories:
                # Find the directory with the highest sequence number
                latest_directory = max(directories, key=lambda d: int(d.split('_')[-1]))
                return os.path.join(self.base_directory, latest_directory)
            print(f"No directories found for the current time: {date_str}. Retrying in 5 seconds.")
            time.sleep(5)

    def get_video_files(self):
        return sorted([f for f in os.listdir(self.directory) if f.endswith('.mp4') or f.endswith('.avi')])

    def play_videos(self):
        face_data = {}
        max_frames = 30
        time_limit = 300
        face_queue = Queue()
        result_queue = Queue()
        analysis_thread = threading.Thread(target=analyze_faces, args=(face_queue, result_queue))
        analysis_thread.setDaemon(True)
        analysis_thread.start()

        while not self.stop:
            if len(self.video_files) < 1:
                self.check_for_new_videos()
                time.sleep(5)
                continue

            if self.current_index >= len(self.video_files):
                self.check_for_new_videos()
                time.sleep(5)
                continue

            video_path = os.path.join(self.directory, self.video_files[self.current_index])
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened() and not self.stop:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                current_time = time.time()

                if results.detections:
                    for idx, detection in enumerate(results.detections):
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        padding = 10
                        x = max(x - padding, 0)
                        y = max(y - padding, 0)
                        w += 2 * padding
                        h += 2 * padding
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        face_id = f"face_{idx}"

                        if face_id not in face_data:
                            face_data[face_id] = {
                              #  'age_values': [],
                              #  'gender_values': [],
                                'emotion_values': [],
                              #  'race_values': [],
                                'frame_count': 0,
                                'locked_label': None,
                                'label_color': (255, 0, 0),
                                'last_seen': current_time
                            }
                        else:
                            face_data[face_id]['last_seen'] = current_time

                        face_frame = frame[y:y + h, x:x + w]

                        face_queue.put((face_id, face_frame))

                        if face_data[face_id]['locked_label']:
                            y_label = y - 10 if y - 10 > 10 else y + h + 20
                            for i, line in enumerate(face_data[face_id]['locked_label'].split(', ')):
                                cv2.putText(frame, line, (x, y_label + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_data[face_id]['label_color'], 2, cv2.LINE_AA)

                while not result_queue.empty():
                    face_id, analysis = result_queue.get()
                    if analysis:
                        result = analysis[0]
                      #  face_data[face_id]['age_values'].append(result['age'])
                       # face_data[face_id]['gender_values'].append(result['dominant_gender'])
                        face_data[face_id]['emotion_values'].append(result['dominant_emotion'])
                       # face_data[face_id]['race_values'].append(result['dominant_race'])

                        face_data[face_id]['frame_count'] += 1

                        if face_data[face_id]['frame_count'] >= max_frames:
                          #  avg_age = sum(face_data[face_id]['age_values']) / len(face_data[face_id]['age_values'])
                           # avg_gender = max(set(face_data[face_id]['gender_values']), key=face_data[face_id]['gender_values'].count)
                            avg_emotion = max(set(face_data[face_id]['emotion_values']), key=face_data[face_id]['emotion_values'].count)
                           # avg_race = max(set(face_data[face_id]['race_values']), key=face_data[face_id]['race_values'].count)

                            known_person = is_known_face(face_frame)
                            if known_person:
                                face_data[face_id]['locked_label'] = f"Name: {known_person}, Emotion: {avg_emotion}"
                                #Age: {avg_age:.1f}, Gender: {avg_gender}, , Race: {avg_race}
                                face_data[face_id]['label_color'] = (255, 0, 0)
                            else:
                                face_data[face_id]['locked_label'] = f"Unknown, Emotion: {avg_emotion}"
                                #Age: {avg_age:.1f}, Gender: {avg_gender}, , Race: {avg_race}
                                face_data[face_id]['label_color'] = (0, 0, 255)

                            #face_data[face_id]['age_values'].clear()
                            #face_data[face_id]['gender_values'].clear()
                            face_data[face_id]['emotion_values'].clear()
                            #face_data[face_id]['race_values'].clear()
                            face_data[face_id]['frame_count'] = 0

                for face_id in list(face_data.keys()):
                    if current_time - face_data[face_id]['last_seen'] > time_limit:
                        del face_data[face_id]

                frame = cv2.resize(frame, (self.window_width, self.window_height))
                img = tk.PhotoImage(master=self.root, data=cv2.imencode('.ppm', frame)[1].tobytes())
                self.label.config(image=img)
                self.label.image = img

                time.sleep(0.03)

            cap.release()
            self.current_index += 1

    def check_for_new_videos(self):
        new_video_files = self.get_video_files()
        if len(new_video_files) > len(self.video_files):
            self.video_files = new_video_files

    def on_closing(self):
        self.stop = True
        self.root.quit()
        self.root.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    base_directory = "/home/wxn/PycharmProjects/MENGProject/output"
    player = VideoPlayer(base_directory, window_width=1600, window_height=1200)  # Adjust the size as needed