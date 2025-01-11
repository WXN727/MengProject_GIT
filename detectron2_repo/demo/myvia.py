import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import csv
import re
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--config-file", default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml", metavar="FILE")
    parser.add_argument("--output", help="Output folder to save CSV and visualizations.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()

    imgOriginalPath = './img/original/'  # Input base folder
    imgDetectionPath = './img/detection/'  # Output base folder

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    # Walk through each action folder (e.g., 1fps_bend_clips_frames)
    for action_folder in os.listdir(imgOriginalPath):
        action_path = os.path.join(imgOriginalPath, action_folder)

        # Process each clip folder (e.g., 30fps_bend_clip_1, 30fps_bend_clip_2)
        for clip_folder in os.listdir(action_path):
            clip_path = os.path.join(action_path, clip_folder)

            # Prepare output directories
            output_clip_dir = os.path.join(imgDetectionPath, action_folder, clip_folder)
            os.makedirs(output_clip_dir, exist_ok=True)

            # Create a CSV file for this clip
            csv_file_path = os.path.join(output_clip_dir, f"{clip_folder}.csv")
            with open(csv_file_path, "w+", encoding="utf-8") as csvFile:
                CSVwriter = csv.writer(csvFile)
                CSVwriter.writerow(
                    ["filename", "file_size", "file_attributes", "region_count",
                     "region_id", "region_shape_attributes", "region_attributes"]
                )

                # Process all images in the clip folder
                # Sort filenames numerically
                img_names = sorted(os.listdir(clip_path), key=lambda x: int(re.search(r'\d+', x).group()))
                for img_name in img_names:

                    img_path = os.path.join(clip_path, img_name)
                    img = read_image(img_path, format="BGR")
                    start_time = time.time()

                    # Run prediction
                    predictions, visualized_output = demo.run_on_image(img)
                    mask = predictions["instances"].pred_classes == 0
                    pred_boxes = predictions["instances"].pred_boxes.tensor[mask]

                    # Log predictions in CSV
                    region_id = 0
                    for box in pred_boxes:
                        box_coords = box.cpu().numpy().tolist()
                        img_region_shape_attributes = {
                            "\"name\"": "\"rect\"",
                            "\"x\"": int(box_coords[0]),
                            "\"y\"": int(box_coords[1]),
                            "\"width\"": int(box_coords[2] - box_coords[0]),
                            "\"height\"": int(box_coords[3] - box_coords[1]),
                        }
                        CSVwriter.writerow(
                            [img_name, os.path.getsize(img_path), '"{}"', len(pred_boxes),
                             region_id, str(img_region_shape_attributes), '"{}"']
                        )
                        region_id += 1

                    # Save visualized output
                    output_img_path = os.path.join(output_clip_dir, img_name)
                    visualized_output.save(output_img_path)

                    logger.info(f"{img_path}: {len(pred_boxes)} people detected in {time.time() - start_time:.2f}s")
