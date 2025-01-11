import torch
import cv2
import numpy as np
from slowfast.config.defaults import get_cfg
from slowfast.models import build_model
from slowfast.utils.checkpoint import load_checkpoint
from slowfast.datasets.utils import pack_pathway_output

# Define default configuration within the script
def get_default_cfg():
    cfg = get_cfg()
    cfg.DATA.NUM_FRAMES = 32
    cfg.DATA.SAMPLING_RATE = 2
    cfg.DATA.TEST_CROP_SIZE = 256
    cfg.DATA.MEAN = [0.45, 0.45, 0.45]
    cfg.DATA.STD = [0.225, 0.225, 0.225]
    cfg.MODEL.ARCH = "slowfast"
    cfg.MODEL.MODEL_NAME = "SlowFast"
    cfg.MODEL.SLOWFAST.BETA_INV = 8
    cfg.MODEL.SLOWFAST.ALPHA = 4
    cfg.MODEL.NUM_CLASSES = 400
    cfg.TEST.CHECKPOINT_FILE_PATH = "path/to/your/checkpoint.pyth"
    cfg.NUM_GPUS = 1
    return cfg

# Function to load the SlowFast model
def load_slowfast_model(cfg):
    model = build_model(cfg)
    model.eval()
    load_checkpoint(cfg.TEST.CHECKPOINT_FILE_PATH, model)
    return model

# Preprocess the frame
def preprocess_frame(frame, cfg):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE))
    frame = frame / 255.0
    frame = frame - np.array(cfg.DATA.MEAN)
    frame = frame / np.array(cfg.DATA.STD)
    frame = frame.transpose(2, 0, 1)
    frame = np.expand_dims(frame, axis=0)
    return frame

# Classify action from frames
def classify_action(model, cfg, frames):
    inputs = pack_pathway_output(cfg, frames)
    inputs = [torch.from_numpy(i).unsqueeze(0).float() for i in inputs]
    with torch.no_grad():
        preds = model(inputs)
    return preds

# Load default configuration
cfg = get_default_cfg()

# Load the model
model = load_slowfast_model(cfg)

# Open the webcam
cap = cv2.VideoCapture(0)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    preprocessed_frame = preprocess_frame(frame, cfg)
    frames.append(preprocessed_frame)

    if len(frames) == cfg.DATA.NUM_FRAMES:
        # Classify action
        preds = classify_action(model, cfg, frames)
        action = preds.argmax(dim=1).item()

        # Map action index to action name (based on the dataset used)
        # This example assumes a simple mapping. Update according to your dataset's classes.
        action_name = ["stand", "sleep", "other_action"][action]  # Simplified example

        # Display the action
        cv2.putText(frame, f'Action: {action_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        frames.pop(0)

    # Display the output
    cv2.imshow('Action Detection', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
