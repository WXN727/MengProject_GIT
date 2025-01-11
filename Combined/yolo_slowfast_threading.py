import torch
import numpy as np
import os, cv2, time, random, pytorchvideo, warnings, argparse, math
from datetime import datetime
import glob
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image, )
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort


class MyVideoCapture:

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.idx = -1
        self.end = False
        self.stack = []

    def read(self):
        self.idx += 1
        ret, img = self.cap.read()
        if ret:
            self.stack.append(img)
        else:
            self.end = True
        return ret, img

    def to_tensor(self, img):
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img.unsqueeze(0)

    def get_video_clip(self):
        assert len(self.stack) > 0, "clip length must be larger than 0!"
        self.stack = [self.to_tensor(img) for img in self.stack]
        clip = torch.cat(self.stack).permute(-1, 0, 1, 2)
        del self.stack
        self.stack = []
        return clip

    def release(self):
        self.cap.release()


def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img


def ava_inference_transform(
        clip,
        boxes,
        num_frames=32,
        crop_size=640,
        data_mean=[0.45, 0.45, 0.45],
        data_std=[0.225, 0.225, 0.225],
        slow_fast_alpha=4,
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes, )
    clip = normalize(clip,
                     np.array(data_mean, dtype=np.float32),
                     np.array(data_std, dtype=np.float32), )
    boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip, 1,
                                          torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), roi_boxes


def plot_one_box(x, img, color=[100, 100, 100], text_info="None",
                 velocity=None, thickness=1, fontsize=0.5, fontthickness=1):
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255, 255, 255], fontthickness)
    return img


def deepsort_update(Tracker, pred, xywh, np_img):
    outputs = Tracker.update(xywh, pred[:, 4:5], pred[:, 5].tolist(), cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    return outputs


def save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, color_map, output_video, vis=False):
    for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
        if pred.shape[0]:
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                if int(cls) != 0:
                    ava_label = ''
                elif trackid in id_to_ava_labels.keys():
                    ava_label = id_to_ava_labels[trackid].split(' ')[0]
                else:
                    ava_label = 'Unknown'
                text = '{} {} {}'.format(int(trackid), yolo_preds.names[int(cls)], ava_label)
                color = color_map[int(cls)]
                im = plot_one_box(box, im, color, text)
        im = im.astype(np.uint8)
        output_video.write(im)
        if vis:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imshow("demo", im)


# Initialize list to store timing data for analytics
timing_data = []

def process_clip(video_file, model, device, imsize, video_model, ava_labelnames, coco_color_map, output_dir, config):
    # Start timing for entire clip processing
    start_time = time.time()
    clip_timing = {'clip': video_file, 'total_time': 0, 'load_time': 0, 'inference_time': 0, 'postprocess_time': 0, 'save_time': 0}

    # Load resources and set up tracker (Load timing)
    load_start = time.time()
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    cap = MyVideoCapture(video_file)
    output_video_path = os.path.join(output_dir, os.path.basename(video_file))
    original_width = int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outputvideo = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (original_width, original_height))
    clip_timing['load_time'] = time.time() - load_start

    # Process frames
    inference_time = 0
    postprocess_time = 0
    save_time = 0

    while not cap.end:
        ret, img = cap.read()
        if not ret:
            continue

        # Inference timing
        inference_start = time.time()
        yolo_preds = model([img], size=imsize)
        inference_time += time.time() - inference_start

        # Post-processing timing
        postprocess_start = time.time()
        # Apply DeepSort and AVA transformations
        deepsort_outputs = []
        for j in range(len(yolo_preds.pred)):
            temp = deepsort_update(deepsort_tracker, yolo_preds.pred[j].cpu(), yolo_preds.xywh[j][:, 0:4].cpu(),
                                   yolo_preds.ims[j])
            if len(temp) == 0:
                temp = np.ones((0, 8))
            deepsort_outputs.append(temp.astype(np.float32))
        yolo_preds.pred = deepsort_outputs
        postprocess_time += time.time() - postprocess_start

        # Save timing
        save_start = time.time()
        save_yolopreds_tovideo(yolo_preds, {}, coco_color_map, outputvideo, config.show)
        save_time += time.time() - save_start

    # Release resources and finalize timing
    cap.release()
    outputvideo.release()
    clip_timing['total_time'] = time.time() - start_time
    clip_timing['inference_time'] = inference_time
    clip_timing['postprocess_time'] = postprocess_time
    clip_timing['save_time'] = save_time

    timing_data.append(clip_timing)
    print(f"Processing time for {video_file}: {clip_timing}")


def main(config):
    device = config.device
    imsize = config.imsize

    model = torch.hub.load('ultralytics/yolov5', 'yolov5l6').to(device)
    model.conf = config.conf
    model.iou = config.iou
    model.max_det = 100
    if config.classes:
        model.classes = config.classes

    video_model = slowfast_r50_detection(True).eval().to(device)

    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    now = datetime.now()
    dir_name_pattern = now.strftime("/home/wxn/PycharmProjects/MENGProject/input/%Y_%m_%d_%H:%M_observation_*")
    print(f"Looking for directories matching: {dir_name_pattern}")
    video_dirs = sorted(glob.glob(dir_name_pattern))

    if not video_dirs:
        print("No video directories found.")
        return

    input_dir = video_dirs[-1]
    print(f"Processing directory: {input_dir}")
    output_dir = input_dir.replace("input", "output").replace("observation", "processed")
    os.makedirs(output_dir, exist_ok=True)

    processed_videos = set()

    while True:
        video_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")) + glob.glob(os.path.join(input_dir, "*.avi")))
        new_videos = [f for f in video_files if f not in processed_videos]

        if not new_videos:
            print("No new video files found, waiting...")
            time.sleep(1)
            continue

        # Process up to 3 clips at a time using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as executor:
            for i in range(0, len(new_videos), 1):
                batch = new_videos[i:i + 1]
                futures = [executor.submit(process_clip, video_file, model, device, imsize, video_model, ava_labelnames, coco_color_map, output_dir, config) for video_file in batch]

                # Wait for all clips in the batch to finish
                for future in futures:
                    future.result()

                # Mark processed videos and clear cache
                processed_videos.update(batch)
                torch.cuda.empty_cache()
                print(torch.cuda.memory_summary())

        # After all clips, save timing data
        df_timing = pd.DataFrame(timing_data)
        df_timing.to_csv("processing_time_analysis.csv", index=False)
        print("Saved timing data to processing_time_analysis.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imsize', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--show', action='store_true', help='show img')
    config = parser.parse_args()

    main(config)
