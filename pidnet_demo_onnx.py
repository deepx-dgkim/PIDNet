#!/usr/bin/env python3
"""PIDNet ONNX demo for pretrained_dynamic.onnx.

Reference implementation: https://github.com/XuJiacong/PIDNet
(see scripts/PIDNet/tools/custom.py for the original pre/post-processing
this demo is based on: ImageNet mean/std normalization in RGB order and
the 19-class Cityscapes color map).

Unlike the upstream repo, pretrained_dynamic.onnx wraps the PIDNet backbone
with an extra ArgMax + Softmax pair (see scripts/export_vinrobotics.py /
scripts/PIDNet/extract_vinrobotics.py) so that the class-index and
per-class-probability post-processing that upstream does on the host after
`model(x)` is instead baked into the graph as two NPU-friendly output
layers:

  - "labels": argmax(logits, dim=1)  -> int64 [1, H/8, W/8]
  - "masks":  softmax(logits, dim=1) -> float32 [1, 19, H/8, W/8]

Both outputs are still at the network's internal stride (1/8th of the
512x512 input), so this demo only has to resize them back up to the
original image size before visualizing -- the per-pixel argmax itself is
no longer something this script needs to compute for the default
"labels" output.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = REPO_ROOT / "pretrained_dynamic.onnx"
NUM_CLASSES = 19
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}

# Same 19 Cityscapes trainId colors as scripts/PIDNet/tools/custom.py, in BGR
# order for OpenCV.
CITYSCAPES_COLORS_BGR = np.array(
    [
        (128, 64, 128),  # road
        (232, 35, 244),  # sidewalk
        (70, 70, 70),  # building
        (156, 102, 102),  # wall
        (153, 153, 190),  # fence
        (153, 153, 153),  # pole
        (30, 170, 250),  # traffic light
        (0, 220, 220),  # traffic sign
        (35, 142, 107),  # vegetation
        (152, 251, 152),  # terrain
        (180, 130, 70),  # sky
        (60, 20, 220),  # person
        (0, 0, 255),  # rider
        (142, 0, 0),  # car
        (70, 0, 0),  # truck
        (100, 60, 0),  # bus
        (100, 80, 0),  # train
        (230, 0, 0),  # motorcycle
        (32, 11, 119),  # bicycle
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load pretrained_dynamic.onnx (PIDNet with ArgMax + Softmax baked "
            "in as output layers) and show segmentation results for one image, "
            "an image folder, or a video file."
        )
    )
    parser.add_argument("input", help="Image file, directory containing images, or video file.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="ONNX model path. Defaults to pretrained_dynamic.onnx in this repo.",
    )
    parser.add_argument(
        "--output",
        choices=("labels", "masks"),
        default="labels",
        help=(
            "Which baked-in output head to visualize. 'labels' is the "
            "in-graph ArgMax result (nearest-neighbor upsampled). 'masks' is "
            "the in-graph Softmax result, upsampled per-class before taking "
            "argmax on the host for smoother boundaries."
        ),
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help="ONNX Runtime providers, e.g. CUDAExecutionProvider CPUExecutionProvider",
    )
    parser.add_argument(
        "--view",
        choices=("overlay", "mask", "side-by-side"),
        default="overlay",
        help="Visualization mode.",
    )
    parser.add_argument("--alpha", type=float, default=0.55, help="Segmentation overlay opacity.")
    parser.add_argument(
        "--folder-delay-ms",
        type=int,
        default=1,
        help="Delay between folder images. Press q or Esc to quit.",
    )
    parser.add_argument(
        "--video-delay-ms",
        type=int,
        default=None,
        help="Delay between video frames. Defaults to the source FPS.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search image folders recursively.")
    parser.add_argument(
        "--max-display-size",
        type=int,
        default=1280,
        help="Resize only the displayed result so its longest side is at most this value. Use 0 to disable.",
    )
    parser.add_argument("--window-name", default="PIDNet ONNX Demo (ArgMax/Softmax)")
    return parser.parse_args()


def import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required to run inference. Install dependencies with "
            "`python3 -m pip install -r requirements.txt` or use this repo's venv."
        ) from exc
    return ort


def choose_providers(requested: list[str] | None) -> list[str]:
    ort = import_onnxruntime()
    if requested:
        return requested
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [provider for provider in preferred if provider in available] or available


def infer_static_input_size(input_shape: list[object]) -> tuple[int, int] | None:
    if len(input_shape) != 4:
        return None
    height, width = input_shape[2], input_shape[3]
    if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0:
        return height, width
    return None


def preprocess(image_bgr: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    height, width = input_size
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_rgb -= IMAGENET_MEAN
    image_rgb /= IMAGENET_STD
    return np.expand_dims(image_rgb.transpose(2, 0, 1), axis=0).astype(np.float32)


def labels_to_prediction(labels: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Upsample the in-graph ArgMax output ("labels") to the original image size."""
    prediction = np.squeeze(labels).astype(np.uint8)
    if prediction.shape == target_hw:
        return prediction
    return cv2.resize(
        prediction, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST
    )


def masks_to_prediction(masks: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Upsample the in-graph Softmax output ("masks") per class, then argmax on host."""
    scores = np.squeeze(masks, axis=0)  # [num_classes, h, w]
    target_h, target_w = target_hw
    if scores.shape[-2:] == target_hw:
        resized = scores
    else:
        resized = np.empty((scores.shape[0], target_h, target_w), dtype=np.float32)
        for class_idx in range(scores.shape[0]):
            resized[class_idx] = cv2.resize(
                scores[class_idx], (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
    return np.argmax(resized, axis=0).astype(np.uint8)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file() and is_image_file(path))


def colorize_prediction(prediction: np.ndarray) -> np.ndarray:
    clipped = np.clip(prediction, 0, len(CITYSCAPES_COLORS_BGR) - 1)
    return CITYSCAPES_COLORS_BGR[clipped]


def make_visualization(image_bgr: np.ndarray, prediction: np.ndarray, view: str, alpha: float) -> np.ndarray:
    color_mask = colorize_prediction(prediction)
    if view == "mask":
        return color_mask
    if view == "side-by-side":
        overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, color_mask, alpha, 0.0)
        return np.hstack((image_bgr, overlay))
    return cv2.addWeighted(image_bgr, 1.0 - alpha, color_mask, alpha, 0.0)


def resize_for_display(image: np.ndarray, max_display_size: int) -> np.ndarray:
    if max_display_size <= 0:
        return image
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_display_size:
        return image
    scale = max_display_size / float(longest)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def set_window_title(window_name: str, title: str) -> None:
    try:
        cv2.setWindowTitle(window_name, title)
    except cv2.error:
        pass


class PIDNetOnnxDemo:
    def __init__(self, args: argparse.Namespace) -> None:
        ort = import_onnxruntime()
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        providers = choose_providers(args.providers)
        self.session = ort.InferenceSession(str(model_path), providers=providers)

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_size = infer_static_input_size(input_meta.shape) or (512, 512)
        self.output_name = args.output

        output_names = {out.name for out in self.session.get_outputs()}
        if self.output_name not in output_names:
            raise ValueError(
                f"Model has no '{self.output_name}' output. Available outputs: {sorted(output_names)}"
            )

        print(f"Model: {model_path}")
        print(f"Providers: {self.session.get_providers()}")
        print(f"Input: {self.input_name}, shape={input_meta.shape}, resized_to={self.input_size}")
        for out in self.session.get_outputs():
            print(f"Output: {out.name}, shape={out.shape}, dtype={out.type}")
        print(f"Visualizing output: {self.output_name}")

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        ort_input = preprocess(image_bgr, self.input_size)
        (output,) = self.session.run([self.output_name], {self.input_name: ort_input})
        target_hw = image_bgr.shape[:2]
        if self.output_name == "labels":
            return labels_to_prediction(output, target_hw)
        return masks_to_prediction(output, target_hw)


def show_result(window_name: str, title: str, result: np.ndarray, max_display_size: int, delay_ms: int) -> int:
    display = resize_for_display(result, max_display_size)
    try:
        cv2.imshow(window_name, display)
        set_window_title(window_name, title)
        return cv2.waitKey(delay_ms) & 0xFF
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV imshow failed. If opencv-python-headless is installed, replace it "
            "with opencv-python in an environment that has GUI display support."
        ) from exc


def create_window(window_name: str) -> None:
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV could not create a GUI window. If opencv-python-headless is installed, "
            "replace it with opencv-python and run in an environment with display support."
        ) from exc


def destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def run_image(demo: PIDNetOnnxDemo, image_path: Path, args: argparse.Namespace, wait_ms: int) -> bool:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    start = time.perf_counter()
    prediction = demo.infer(image)
    elapsed = time.perf_counter() - start
    result = make_visualization(image, prediction, args.view, args.alpha)
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    key = show_result(
        args.window_name,
        f"{args.window_name} - {image_path.name} - {fps:.1f} FPS",
        result,
        args.max_display_size,
        wait_ms,
    )
    return key not in (27, ord("q"), ord("Q"))


def run_image_folder(demo: PIDNetOnnxDemo, input_dir: Path, args: argparse.Namespace) -> None:
    image_paths = find_images(input_dir, args.recursive)
    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {input_dir}")

    print(f"Found {len(image_paths)} image(s). Press q or Esc in the display window to quit.")
    for image_path in image_paths:
        if not run_image(demo, image_path, args, max(1, args.folder_delay_ms)):
            break


def video_delay_ms(video: cv2.VideoCapture, requested: int | None) -> int:
    if requested is not None:
        return max(1, requested)
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return max(1, int(round(1000.0 / fps)))
    return 1


def run_video(demo: PIDNetOnnxDemo, video_path: Path, args: argparse.Namespace) -> None:
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    delay_ms = video_delay_ms(video, args.video_delay_ms)
    frame_index = 0
    print("Press q or Esc in the display window to quit.")

    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break

            frame_index += 1
            start = time.perf_counter()
            prediction = demo.infer(frame)
            elapsed = time.perf_counter() - start
            result = make_visualization(frame, prediction, args.view, args.alpha)
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            key = show_result(
                args.window_name,
                f"{args.window_name} - {video_path.name} - frame {frame_index} - {fps:.1f} FPS",
                result,
                args.max_display_size,
                delay_ms,
            )
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        video.release()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    demo = PIDNetOnnxDemo(args)
    create_window(args.window_name)

    try:
        if input_path.is_dir():
            run_image_folder(demo, input_path, args)
        elif is_image_file(input_path):
            run_image(demo, input_path, args, wait_ms=0)
        elif is_video_file(input_path):
            run_video(demo, input_path, args)
        else:
            raise ValueError(
                f"Unsupported input type: {input_path}. Use an image file, image directory, or video file."
            )
    finally:
        destroy_windows()


if __name__ == "__main__":
    main()
