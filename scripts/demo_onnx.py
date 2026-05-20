#!/usr/bin/env python3
"""Run a real-time OpenCV demo with a PIDNet Cityscapes ONNX model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = REPO_ROOT / "pidnet_s_cityscapes_val.onnx"

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}

CITYSCAPES_COLORS = np.array(
    [
        (128, 64, 128),  # road
        (244, 35, 232),  # sidewalk
        (70, 70, 70),  # building
        (102, 102, 156),  # wall
        (190, 153, 153),  # fence
        (153, 153, 153),  # pole
        (250, 170, 30),  # traffic light
        (220, 220, 0),  # traffic sign
        (107, 142, 35),  # vegetation
        (152, 251, 152),  # terrain
        (70, 130, 180),  # sky
        (220, 20, 60),  # person
        (255, 0, 0),  # rider
        (0, 0, 142),  # car
        (0, 0, 70),  # truck
        (0, 60, 100),  # bus
        (0, 80, 100),  # train
        (0, 0, 230),  # motorcycle
        (119, 11, 32),  # bicycle
    ],
    dtype=np.uint8,
)
CITYSCAPES_COLORS_BGR = CITYSCAPES_COLORS[:, ::-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load pidnet_s_cityscapes_val.onnx and show segmentation results for "
            "one image, an image folder, or a video file."
        )
    )
    parser.add_argument(
        "input",
        help="Image file, directory containing images, or video file.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="ONNX model path. Defaults to pidnet_s_cityscapes_val.onnx in this repo.",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Resize input before inference. Defaults to the ONNX static input size.",
    )
    parser.add_argument(
        "--output-index",
        type=int,
        default=-1,
        help="Output tensor index when the ONNX graph has multiple outputs.",
    )
    parser.add_argument("--num-classes", type=int, default=19)
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Segmentation overlay opacity for --view overlay.",
    )
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
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search image folders recursively.",
    )
    parser.add_argument(
        "--max-display-size",
        type=int,
        default=1280,
        help="Resize only the displayed result so its longest side is at most this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--window-name",
        default="PIDNet ONNX Segmentation",
        help="OpenCV imshow window name.",
    )
    return parser.parse_args()


def import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required to run inference. Install dependencies with "
            "`python3 -m pip install -r requirements.txt` or use this repo's .venv."
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


def preprocess(image_bgr: np.ndarray, input_size: tuple[int, int] | None) -> np.ndarray:
    if input_size is not None:
        height, width = input_size
        image_bgr = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

    image_rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
    image_rgb -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    image_rgb /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.expand_dims(image_rgb.transpose(2, 0, 1), axis=0).astype(np.float32)


def resize_scores(scores: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if scores.shape[-2:] == target_hw:
        return scores

    resized = np.empty((scores.shape[0], target_h, target_w), dtype=np.float32)
    for class_idx in range(scores.shape[0]):
        resized[class_idx] = cv2.resize(
            scores[class_idx], (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
    return resized


def output_to_prediction(
    output: np.ndarray,
    target_hw: tuple[int, int],
    num_classes: int,
) -> np.ndarray:
    if output.ndim == 4 and output.shape[1] == num_classes:
        scores = resize_scores(output[0], target_hw)
        return np.argmax(scores, axis=0).astype(np.uint8)

    if output.ndim == 4 and output.shape[-1] == num_classes:
        scores = output[0].transpose(2, 0, 1)
        scores = resize_scores(scores, target_hw)
        return np.argmax(scores, axis=0).astype(np.uint8)

    if output.ndim == 3 and output.shape[0] == num_classes:
        scores = resize_scores(output, target_hw)
        return np.argmax(scores, axis=0).astype(np.uint8)

    if output.ndim == 3 and output.shape[-1] == num_classes:
        scores = output.transpose(2, 0, 1)
        scores = resize_scores(scores, target_hw)
        return np.argmax(scores, axis=0).astype(np.uint8)

    prediction = np.squeeze(output)
    if prediction.ndim != 2:
        raise ValueError(f"Unsupported ONNX output shape: {output.shape}")
    if prediction.shape != target_hw:
        prediction = cv2.resize(
            prediction.astype(np.uint8),
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return prediction.astype(np.uint8)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and is_image_file(path)
    )


def colorize_prediction(prediction: np.ndarray) -> np.ndarray:
    clipped = np.clip(prediction, 0, len(CITYSCAPES_COLORS_BGR) - 1)
    return CITYSCAPES_COLORS_BGR[clipped]


def make_visualization(
    image_bgr: np.ndarray,
    prediction: np.ndarray,
    view: str,
    alpha: float,
) -> np.ndarray:
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
        self.input_size = (
            tuple(args.input_size)
            if args.input_size
            else infer_static_input_size(input_meta.shape)
        )
        self.output_index = args.output_index
        self.num_classes = args.num_classes

        print(f"Model: {model_path}")
        print(f"Providers: {self.session.get_providers()}")
        print(
            f"Input: {self.input_name}, shape={input_meta.shape}, "
            f"demo_size={self.input_size}"
        )

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        ort_input = preprocess(image_bgr, self.input_size)
        outputs = self.session.run(None, {self.input_name: ort_input})
        selected = outputs[self.output_index]
        return output_to_prediction(selected, image_bgr.shape[:2], self.num_classes)


def show_result(
    window_name: str,
    title: str,
    result: np.ndarray,
    max_display_size: int,
    delay_ms: int,
) -> int:
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


def run_image(
    demo: PIDNetOnnxDemo,
    image_path: Path,
    args: argparse.Namespace,
    wait_ms: int,
) -> bool:
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
