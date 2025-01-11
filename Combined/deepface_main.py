import subprocess
import os
import cv2
import numpy as np
from deepface import DeepFace


def run_yolo_slowfast(input_path, output_path):
    command = f"python yolo_slowfast.py --input {input_path} --output {output_path}"
    try:
        process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.stdout.decode()
        print("YOLO SlowFast command executed successfully:")
        print(output)
    except subprocess.CalledProcessError as e:
        print("Error occurred while executing the YOLO SlowFast command:")
        print(e.stderr.decode())


def annotate_frame(frame, analysis, box):
    x, y, w, h = box

    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Extract information from the analysis
    emotion = analysis['dominant_emotion']
    emotion_conf = analysis['emotion'][emotion]
    age = analysis['age']
    gender = analysis['gender']
    race = analysis['dominant_race']
    race_conf = analysis['race'][race]

    # Define the text to display
    text = f"Emotion: {emotion} ({emotion_conf:.2f}%)\nAge: {age}\nGender: {gender}\nRace: {race} ({race_conf:.2f}%)"
    y0, dy = y - 100, 30
    for i, line in enumerate(text.split('\n')):
        y_line = y0 + i * dy
        cv2.putText(frame, line, (x, y_line), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


def analyze_and_annotate_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return

    print(f"Processing video: {input_path}")
    print(f"Output video will be saved to: {output_path}")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        try:
            # Analyze the frame with DeepFace
            analysis = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)
            if 'region' in analysis and analysis['region'] is not None:
                for face_info in analysis['region']:
                    x, y, w, h = face_info['x'], face_info['y'], face_info['w'], face_info['h']
                    frame = annotate_frame(frame, analysis, (x, y, w, h))
        except Exception as e:
            print(f"Error analyzing frame {frame_count}: {e}")

        out.write(frame)

    cap.release()
    out.release()
    print(f"Annotated video saved at {output_path}")


if __name__ == "__main__":
    input_path = '/home/wxn/PycharmProjects/MENGProject/input/video_20240722-10:48:21.avi'
    yolo_output_path = '/home/wxn/PycharmProjects/MENGProject/output/output_video_yolo.avi'
    annotated_output_path = '/home/wxn/PycharmProjects/MENGProject/output/output_video_annotated.avi'

    print(f"Current working directory: {os.getcwd()}")
    print(f"Input path: {input_path}")
    print(f"YOLO output path: {yolo_output_path}")
    print(f"Annotated output path: {annotated_output_path}")

    run_yolo_slowfast(input_path, yolo_output_path)

    # Ensure the YOLO output file exists before running DeepFace
    if os.path.exists(yolo_output_path):
        analyze_and_annotate_video(yolo_output_path, annotated_output_path)
    else:
        print(f"Output video not found at {yolo_output_path}. DeepFace will not be run.")
