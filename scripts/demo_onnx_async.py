#!/usr/bin/env python3
"""Run an asynchronous OpenCV video demo with a PIDNet Cityscapes ONNX model."""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demo_onnx import (
    PIDNetOnnxDemo,
    create_window,
    destroy_windows,
    is_image_file,
    is_video_file,
    make_visualization,
    resize_for_display,
    run_image,
    run_image_folder,
    set_window_title,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "pidnet_s_cityscapes_val.onnx"
QUIT_KEYS = {27, ord("q"), ord("Q")}
STOP = object()


@dataclass(frozen=True)
class VideoFrame:
    index: int
    image_bgr: np.ndarray
    captured_at: float


@dataclass(frozen=True)
class VideoResult:
    frame_index: int
    image_bgr: np.ndarray
    prediction: np.ndarray
    infer_ms: float
    captured_at: float
    finished_at: float


@dataclass(frozen=True)
class StatsSnapshot:
    captured: int
    inferred: int
    displayed: int
    dropped_inputs: int
    dropped_results: int
    skipped_results: int
    total_infer_time: float
    start_time: float


@dataclass
class AsyncStats:
    start_time: float = field(default_factory=time.perf_counter)
    captured: int = 0
    inferred: int = 0
    displayed: int = 0
    dropped_inputs: int = 0
    dropped_results: int = 0
    skipped_results: int = 0
    total_infer_time: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_captured(self) -> None:
        with self.lock:
            self.captured += 1

    def add_inferred(self, elapsed: float) -> None:
        with self.lock:
            self.inferred += 1
            self.total_infer_time += elapsed

    def add_displayed(self) -> None:
        with self.lock:
            self.displayed += 1

    def add_dropped_inputs(self, count: int = 1) -> None:
        with self.lock:
            self.dropped_inputs += count

    def add_dropped_results(self, count: int = 1) -> None:
        with self.lock:
            self.dropped_results += count

    def add_skipped_results(self, count: int = 1) -> None:
        if count <= 0:
            return
        with self.lock:
            self.skipped_results += count

    def snapshot(self) -> StatsSnapshot:
        with self.lock:
            return StatsSnapshot(
                captured=self.captured,
                inferred=self.inferred,
                displayed=self.displayed,
                dropped_inputs=self.dropped_inputs,
                dropped_results=self.dropped_results,
                skipped_results=self.skipped_results,
                total_infer_time=self.total_infer_time,
                start_time=self.start_time,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load pidnet_s_cityscapes_val.onnx and show segmentation results for "
            "one image, an image folder, or a video file. Video input uses an async "
            "capture/inference/preview pipeline."
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
        "--preview-delay-ms",
        type=int,
        default=1,
        help="Delay used by the async preview loop. Use 1 for maximum video preview FPS.",
    )
    parser.add_argument(
        "--video-delay-ms",
        type=int,
        default=None,
        help="Compatibility alias for --preview-delay-ms when running video input.",
    )
    parser.add_argument(
        "--inference-workers",
        type=int,
        default=0,
        help="Number of ONNX Runtime worker threads for video. Use 0 for auto.",
    )
    parser.add_argument(
        "--frame-queue-size",
        type=int,
        default=4,
        help="Maximum decoded video frames waiting for inference.",
    )
    parser.add_argument(
        "--result-queue-size",
        type=int,
        default=4,
        help="Maximum rendered inference results waiting for preview.",
    )
    parser.add_argument(
        "--drop-input-frames",
        action="store_true",
        help="Drop old decoded frames when inference is behind. Useful for low-latency streams.",
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
        help=(
            "Resize only the displayed result so its longest side is at most this value. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--window-name",
        default="PIDNet ONNX Async Segmentation",
        help="OpenCV imshow window name.",
    )
    return parser.parse_args()


def resolve_inference_workers(requested: int, providers: list[str]) -> int:
    if requested > 0:
        return requested
    cpu_count = os.cpu_count() or 1
    gpu_provider_tokens = ("CUDA", "Tensorrt", "Dml", "ROCM", "OpenVINO")
    if any(any(token in provider for token in gpu_provider_tokens) for provider in providers):
        return 2
    return max(1, min(2, cpu_count // 2 or 1))


def validate_queue_size(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value


def preview_delay_ms(args: argparse.Namespace) -> int:
    requested = args.video_delay_ms if args.video_delay_ms is not None else args.preview_delay_ms
    return max(1, requested)


def put_with_backpressure(
    items: queue.Queue[Any],
    item: Any,
    stop_event: threading.Event,
) -> bool:
    while not stop_event.is_set():
        try:
            items.put(item, timeout=0.05)
            return True
        except queue.Full:
            continue
    return False


def put_latest(
    items: queue.Queue[Any],
    item: Any,
    stop_event: threading.Event,
    on_drop: Any | None = None,
) -> bool:
    while not stop_event.is_set():
        try:
            items.put(item, timeout=0.01)
            return True
        except queue.Full:
            try:
                items.get_nowait()
            except queue.Empty:
                continue
            if on_drop is not None:
                on_drop()
    return False


def force_put_latest(items: queue.Queue[Any], item: Any, on_drop: Any | None = None) -> None:
    while True:
        try:
            items.put_nowait(item)
            return
        except queue.Full:
            try:
                items.get_nowait()
            except queue.Empty:
                continue
            if on_drop is not None:
                on_drop()


def read_video_frames(
    video_path: Path,
    frame_queue: queue.Queue[Any],
    errors: queue.Queue[BaseException],
    stats: AsyncStats,
    worker_count: int,
    stop_event: threading.Event,
    drop_input_frames: bool,
) -> None:
    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_index = 0
        while not stop_event.is_set():
            ok, frame = video.read()
            if not ok:
                break

            frame_index += 1
            stats.add_captured()
            item = VideoFrame(frame_index, frame, time.perf_counter())
            if drop_input_frames:
                if not put_latest(
                    frame_queue,
                    item,
                    stop_event,
                    on_drop=stats.add_dropped_inputs,
                ):
                    break
            elif not put_with_backpressure(frame_queue, item, stop_event):
                break
    except BaseException as exc:
        errors.put(exc)
        stop_event.set()
    finally:
        video.release()
        for _ in range(worker_count):
            if stop_event.is_set():
                force_put_latest(frame_queue, STOP)
            else:
                put_with_backpressure(frame_queue, STOP, stop_event)


def run_inference_worker(
    worker_id: int,
    demo: PIDNetOnnxDemo,
    args: argparse.Namespace,
    frame_queue: queue.Queue[Any],
    result_queue: queue.Queue[VideoResult],
    errors: queue.Queue[BaseException],
    stats: AsyncStats,
    stop_event: threading.Event,
) -> None:
    del worker_id
    try:
        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item is STOP:
                break
            if not isinstance(item, VideoFrame):
                continue

            start = time.perf_counter()
            prediction = demo.infer(item.image_bgr)
            elapsed = time.perf_counter() - start
            stats.add_inferred(elapsed)

            video_result = VideoResult(
                frame_index=item.index,
                image_bgr=item.image_bgr,
                prediction=prediction,
                infer_ms=elapsed * 1000.0,
                captured_at=item.captured_at,
                finished_at=time.perf_counter(),
            )
            put_latest(
                result_queue,
                video_result,
                stop_event,
                on_drop=stats.add_dropped_results,
            )
    except BaseException as exc:
        errors.put(exc)
        stop_event.set()


def raise_thread_error(errors: queue.Queue[BaseException]) -> None:
    try:
        error = errors.get_nowait()
    except queue.Empty:
        return
    raise error


def drain_latest_result(
    result_queue: queue.Queue[VideoResult],
    last_displayed_frame: int,
) -> tuple[VideoResult | None, int]:
    latest: VideoResult | None = None
    drained = 0

    while True:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            break

        drained += 1
        if result.frame_index <= last_displayed_frame:
            continue
        if latest is None or result.frame_index > latest.frame_index:
            latest = result

    if latest is None:
        return None, drained
    return latest, drained - 1


def wait_for_preview_key(delay_ms: int) -> int:
    try:
        return cv2.waitKey(delay_ms) & 0xFF
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV waitKey failed. If opencv-python-headless is installed, replace it "
            "with opencv-python in an environment that has GUI display support."
        ) from exc


def show_async_result(
    window_name: str,
    title: str,
    display: np.ndarray,
    delay_ms: int,
) -> int:
    try:
        cv2.imshow(window_name, display)
        set_window_title(window_name, title)
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV imshow failed. If opencv-python-headless is installed, replace it "
            "with opencv-python in an environment that has GUI display support."
        ) from exc
    return wait_for_preview_key(delay_ms)


def add_status_overlay_top_right(image: np.ndarray, lines: list[str]) -> np.ndarray:
    if not lines:
        return image

    output = image.copy()
    height, width = output.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.75, width / 1600.0))
    thickness = max(1, int(round(font_scale * 2.0)))
    padding = max(6, int(round(10 * font_scale)))
    margin = max(8, int(round(14 * font_scale)))
    line_gap = max(4, int(round(7 * font_scale)))

    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    box_width = max(size[0] for size in sizes) + padding * 2
    box_height = sum(size[1] for size in sizes) + line_gap * (len(lines) - 1) + padding * 2
    x0 = max(0, width - box_width - margin)
    y0 = margin
    x1 = min(width - 1, width - margin)
    y1 = min(height - 1, y0 + box_height)

    shaded = output.copy()
    cv2.rectangle(shaded, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(shaded, 0.58, output, 0.42, 0.0, output)

    y = y0 + padding
    for idx, line in enumerate(lines):
        text_height = sizes[idx][1]
        y += text_height
        color = (80, 255, 120) if idx == 0 else (245, 245, 245)
        cv2.putText(
            output,
            line,
            (x0 + padding, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_gap
    return output


def metrics_for_result(
    result: VideoResult,
    stats: AsyncStats,
    worker_count: int,
) -> tuple[float, float, float, int]:
    del worker_count
    snapshot = stats.snapshot()
    elapsed = max(time.perf_counter() - snapshot.start_time, 1e-9)
    infer_fps = snapshot.inferred / elapsed
    display_fps = snapshot.displayed / elapsed
    latency_ms = (time.perf_counter() - result.captured_at) * 1000.0
    dropped = snapshot.dropped_inputs + snapshot.dropped_results + snapshot.skipped_results
    return infer_fps, display_fps, latency_ms, dropped


def overlay_lines_for_result(
    result: VideoResult,
    stats: AsyncStats,
    worker_count: int,
) -> list[str]:
    infer_fps, display_fps, latency_ms, dropped = metrics_for_result(
        result,
        stats,
        worker_count,
    )
    return [
        f"Infer {infer_fps:.1f} FPS",
        f"Preview {display_fps:.1f} FPS",
        f"Latency {latency_ms:.0f} ms",
        f"Dropped {dropped}",
    ]


def title_for_result(
    window_name: str,
    video_path: Path,
    result: VideoResult,
    stats: AsyncStats,
    worker_count: int,
) -> str:
    infer_fps, display_fps, latency_ms, dropped = metrics_for_result(
        result,
        stats,
        worker_count,
    )
    return (
        f"{window_name} - {video_path.name} - frame {result.frame_index} - "
        f"{infer_fps:.1f} infer FPS / {display_fps:.1f} preview FPS - "
        f"{result.infer_ms:.1f} ms - {latency_ms:.1f} ms latency - "
        f"workers {worker_count} - dropped {dropped}"
    )


def print_final_stats(stats: AsyncStats) -> None:
    snapshot = stats.snapshot()
    elapsed = max(time.perf_counter() - snapshot.start_time, 1e-9)
    infer_fps = snapshot.inferred / elapsed
    display_fps = snapshot.displayed / elapsed
    avg_infer_ms = (
        (snapshot.total_infer_time / snapshot.inferred) * 1000.0
        if snapshot.inferred
        else 0.0
    )
    print(
        "Async video summary: "
        f"captured={snapshot.captured}, inferred={snapshot.inferred}, "
        f"displayed={snapshot.displayed}, infer_fps={infer_fps:.2f}, "
        f"preview_fps={display_fps:.2f}, avg_infer_ms={avg_infer_ms:.2f}, "
        f"dropped_input={snapshot.dropped_inputs}, "
        f"dropped_result={snapshot.dropped_results}, "
        f"skipped_preview={snapshot.skipped_results}"
    )


def run_video_async(demo: PIDNetOnnxDemo, video_path: Path, args: argparse.Namespace) -> None:
    worker_count = resolve_inference_workers(
        args.inference_workers,
        demo.session.get_providers(),
    )
    frame_queue_size = validate_queue_size(args.frame_queue_size, "--frame-queue-size")
    result_queue_size = validate_queue_size(args.result_queue_size, "--result-queue-size")
    delay_ms = preview_delay_ms(args)

    frame_queue: queue.Queue[Any] = queue.Queue(maxsize=frame_queue_size)
    result_queue: queue.Queue[VideoResult] = queue.Queue(maxsize=result_queue_size)
    errors: queue.Queue[BaseException] = queue.Queue()
    stats = AsyncStats()
    stop_event = threading.Event()

    reader = threading.Thread(
        target=read_video_frames,
        name="video-reader",
        args=(
            video_path,
            frame_queue,
            errors,
            stats,
            worker_count,
            stop_event,
            args.drop_input_frames,
        ),
        daemon=True,
    )
    workers = [
        threading.Thread(
            target=run_inference_worker,
            name=f"onnx-worker-{idx + 1}",
            args=(
                idx,
                demo,
                args,
                frame_queue,
                result_queue,
                errors,
                stats,
                stop_event,
            ),
            daemon=True,
        )
        for idx in range(worker_count)
    ]

    print(
        "Async video pipeline: "
        f"workers={worker_count}, frame_queue={frame_queue_size}, "
        f"result_queue={result_queue_size}, preview_delay_ms={delay_ms}, "
        f"drop_input_frames={args.drop_input_frames}"
    )
    print("Press q or Esc in the display window to quit.")

    reader.start()
    for worker in workers:
        worker.start()

    last_displayed_frame = 0
    threads = [reader, *workers]
    try:
        while True:
            raise_thread_error(errors)
            latest, skipped = drain_latest_result(result_queue, last_displayed_frame)
            stats.add_skipped_results(skipped)

            if latest is not None:
                last_displayed_frame = latest.frame_index
                stats.add_displayed()
                result = make_visualization(
                    latest.image_bgr,
                    latest.prediction,
                    args.view,
                    args.alpha,
                )
                display = resize_for_display(result, args.max_display_size)
                display = add_status_overlay_top_right(
                    display,
                    overlay_lines_for_result(latest, stats, worker_count),
                )
                key = show_async_result(
                    args.window_name,
                    title_for_result(
                        args.window_name,
                        video_path,
                        latest,
                        stats,
                        worker_count,
                    ),
                    display,
                    delay_ms,
                )
            else:
                key = wait_for_preview_key(delay_ms)

            if key in QUIT_KEYS:
                break
            if not any(thread.is_alive() for thread in threads) and result_queue.empty():
                break
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()
        raise_thread_error(errors)
        print_final_stats(stats)


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
            run_video_async(demo, input_path, args)
        else:
            raise ValueError(
                f"Unsupported input type: {input_path}. "
                "Use an image file, image directory, or video file."
            )
    finally:
        destroy_windows()


if __name__ == "__main__":
    main()
