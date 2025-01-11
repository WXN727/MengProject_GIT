import torch
import numpy as np
import os, cv2, time, random, pytorchvideo, warnings, argparse, math
from datetime import datetime
import glob

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
        num_frames=32,  # if using slowfast_r50_detection, change this to 32, 4 for slow
        crop_size=640,
        data_mean=[0.45, 0.45, 0.45],
        data_std=[0.225, 0.225, 0.225],
        slow_fast_alpha=4,  # if using slowfast_r50_detection, change this to 4, None for slow
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
    # Plots one bounding box on image img
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

    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    # Lock the directory based on the date and time when the code is executed
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

        for video_file in new_videos:
            video_filename = os.path.basename(video_file)
            output_video_path = os.path.join(output_dir, video_filename)

            cap = MyVideoCapture(video_file)
            id_to_ava_labels = {}

            # Retrieve original video dimensions
            original_width = int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            outputvideo = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (original_width, original_height))

            print(f"Processing {video_file}...")

            a = time.time()
            while not cap.end:
                ret, img = cap.read()
                if not ret:
                    continue
                yolo_preds = model([img], size=imsize)
                yolo_preds.files = ["img.jpg"]

                deepsort_outputs = []
                for j in range(len(yolo_preds.pred)):
                    temp = deepsort_update(deepsort_tracker, yolo_preds.pred[j].cpu(), yolo_preds.xywh[j][:, 0:4].cpu(),
                                           yolo_preds.ims[j])
                    if len(temp) == 0:
                        temp = np.ones((0, 8))
                    deepsort_outputs.append(temp.astype(np.float32))

                yolo_preds.pred = deepsort_outputs

                if len(cap.stack) == 25:
                    print(f"Processing {cap.idx // 25}th second clips")
                    clip = cap.get_video_clip()
                    if yolo_preds.pred[0].shape[0]:
                        inputs, inp_boxes, _ = ava_inference_transform(clip, yolo_preds.pred[0][:, 0:4], crop_size=imsize)
                        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
                        if isinstance(inputs, list):
                            inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                        else:
                            inputs = inputs.unsqueeze(0).to(device)
                        with torch.no_grad():
                            slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                            slowfaster_preds = slowfaster_preds.cpu()
                        for tid, avalabel in zip(yolo_preds.pred[0][:, 5].tolist(),
                                                 np.argmax(slowfaster_preds, axis=1).tolist()):
                            id_to_ava_labels[tid] = ava_labelnames[avalabel + 1]

                save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, coco_color_map, outputvideo, config.show)

            print("Total cost: {:.3f} s, video length: {} s".format(time.time() - a, cap.idx / 25))

            cap.release()
            outputvideo.release()
            print(f'Saved video to: {output_video_path}')
            processed_videos.add(video_file)


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

