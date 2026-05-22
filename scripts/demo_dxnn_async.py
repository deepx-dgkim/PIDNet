#!/usr/bin/env python3
"""Run an asynchronous OpenCV video demo with a PIDNet Cityscapes DXNN model."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dx_engine import InferenceEngine, InferenceOption

from demo_dxnn import (
    DEFAULT_MODEL,
    PIDNetDxnnDemo,
    create_window,
    destroy_windows,
    dtype_name,
    infer_input_layout,
    infer_static_input_size,
    is_image_file,
    is_video_file,
    make_visualization,
    output_to_prediction,
    preprocess_dxnn,
    reshape_output_if_needed,
    resize_for_display,
    run_image,
    run_image_folder,
    set_window_title,
)


QUIT_KEYS = {27, ord("q"), ord("Q")}
STOP = object()

BOUND_OPTIONS = {
    "npu_all": InferenceOption.BOUND_OPTION.NPU_ALL,
    "npu_0": InferenceOption.BOUND_OPTION.NPU_0,
    "npu_1": InferenceOption.BOUND_OPTION.NPU_1,
    "npu_2": InferenceOption.BOUND_OPTION.NPU_2,
    "npu_01": InferenceOption.BOUND_OPTION.NPU_01,
    "npu_12": InferenceOption.BOUND_OPTION.NPU_12,
    "npu_02": InferenceOption.BOUND_OPTION.NPU_02,
}


@dataclass(frozen=True)
class CapturedFrame:
    index: int
    image_bgr: np.ndarray
    captured_at: float


@dataclass(frozen=True)
class SubmittedJob:
    frame_index: int
    image_bgr: np.ndarray
    input_tensor: np.ndarray
    req_id: int
    captured_at: float
    submitted_at: float


@dataclass(frozen=True)
class CompletedJob:
    frame_index: int
    image_bgr: np.ndarray
    outputs: list[np.ndarray]
    captured_at: float
    submitted_at: float
    completed_at: float


@dataclass(frozen=True)
class PredictionResult:
    frame_index: int
    image_bgr: np.ndarray
    prediction: np.ndarray
    captured_at: float
    submitted_at: float
    completed_at: float
    postprocessed_at: float
    async_latency_ms: float
    postprocess_ms: float


@dataclass(frozen=True)
class StatsSnapshot:
    captured: int
    submitted: int
    completed: int
    postprocessed: int
    displayed: int
    dropped_inputs: int
    dropped_outputs: int
    dropped_results: int
    skipped_results: int
    sum_preprocess: float
    sum_async_latency: float
    sum_postprocess: float
    inflight_current: int
    inflight_max: int
    first_submit_at: float | None
    last_complete_at: float | None
    start_time: float


@dataclass
class AsyncStats:
    start_time: float = field(default_factory=time.perf_counter)
    captured: int = 0
    submitted: int = 0
    completed: int = 0
    postprocessed: int = 0
    displayed: int = 0
    dropped_inputs: int = 0
    dropped_outputs: int = 0
    dropped_results: int = 0
    skipped_results: int = 0
    sum_preprocess: float = 0.0
    sum_async_latency: float = 0.0
    sum_postprocess: float = 0.0
    inflight_current: int = 0
    inflight_max: int = 0
    first_submit_at: float | None = None
    last_complete_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_captured(self) -> None:
        with self.lock:
            self.captured += 1

    def add_submitted(self, preprocess_elapsed: float, submitted_at: float) -> None:
        with self.lock:
            self.submitted += 1
            self.sum_preprocess += preprocess_elapsed
            self.inflight_current += 1
            self.inflight_max = max(self.inflight_max, self.inflight_current)
            if self.first_submit_at is None:
                self.first_submit_at = submitted_at

    def add_completed(self, async_elapsed: float, completed_at: float) -> None:
        with self.lock:
            self.completed += 1
            self.sum_async_latency += async_elapsed
            self.last_complete_at = completed_at
            self.inflight_current = max(0, self.inflight_current - 1)

    def add_postprocessed(self, elapsed: float) -> None:
        with self.lock:
            self.postprocessed += 1
            self.sum_postprocess += elapsed

    def add_displayed(self) -> None:
        with self.lock:
            self.displayed += 1

    def add_dropped_inputs(self, count: int = 1) -> None:
        with self.lock:
            self.dropped_inputs += count

    def add_dropped_outputs(self, count: int = 1) -> None:
        with self.lock:
            self.dropped_outputs += count

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
                submitted=self.submitted,
                completed=self.completed,
                postprocessed=self.postprocessed,
                displayed=self.displayed,
                dropped_inputs=self.dropped_inputs,
                dropped_outputs=self.dropped_outputs,
                dropped_results=self.dropped_results,
                skipped_results=self.skipped_results,
                sum_preprocess=self.sum_preprocess,
                sum_async_latency=self.sum_async_latency,
                sum_postprocess=self.sum_postprocess,
                inflight_current=self.inflight_current,
                inflight_max=self.inflight_max,
                first_submit_at=self.first_submit_at,
                last_complete_at=self.last_complete_at,
                start_time=self.start_time,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load pidnet_s_cityscapes_val.dxnn and show segmentation results for "
            "one image, an image folder, or a video file. Video input uses DXRT "
            "run_async/wait to keep the NPU saturated."
        )
    )
    parser.add_argument(
        "input",
        help="Image file, directory containing images, or video file.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="DXNN model path. Defaults to dxnn/pidnet_s_cityscapes_val.dxnn in this repo.",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Resize input before inference. Defaults to the DXNN static input size.",
    )
    parser.add_argument(
        "--input-color",
        choices=("bgr", "rgb"),
        default="bgr",
        help=(
            "Color order sent to DXRT. Defaults to bgr for DXNNs compiled with "
            "BGR2RGB preprocessing."
        ),
    )
    parser.add_argument(
        "--input-layout",
        choices=("auto", "nhwc", "nchw"),
        default="auto",
        help="Input tensor layout. Defaults to auto from DXNN input shape.",
    )
    parser.add_argument(
        "--output-index",
        type=int,
        default=-1,
        help="Output tensor index when the DXNN graph has multiple outputs.",
    )
    parser.add_argument("--num-classes", type=int, default=19)
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
        "--max-inflight",
        type=int,
        default=0,
        help=(
            "Maximum DXRT async jobs in flight. Use 0 to match "
            "InferenceOption.buffer_count."
        ),
    )
    parser.add_argument(
        "--buffer-count",
        type=int,
        default=0,
        help="DXRT InferenceOption buffer_count. Use 0 for DXRT default.",
    )
    parser.add_argument(
        "--bound-option",
        choices=tuple(BOUND_OPTIONS),
        default="npu_all",
        help="NPU core binding. Defaults to npu_all.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        type=int,
        default=None,
        help="DXRT device IDs to use, e.g. --devices 0 1.",
    )
    parser.add_argument(
        "--disable-ort",
        action="store_true",
        help="Force InferenceOption.use_ort=False for pure DXRT/NPU execution where supported.",
    )
    parser.add_argument(
        "--frame-queue-size",
        type=int,
        default=0,
        help="Decoded frame queue size. Use 0 for max(2 * max_inflight, 8).",
    )
    parser.add_argument(
        "--output-queue-size",
        type=int,
        default=0,
        help="Completed DXRT output queue size. Use 0 for max_inflight.",
    )
    parser.add_argument(
        "--result-queue-size",
        type=int,
        default=4,
        help="Maximum postprocessed results waiting for preview.",
    )
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        default=2,
        help="CPU postprocess workers. Increase if postprocess throttles async wait.",
    )
    parser.add_argument(
        "--drop-input-frames",
        action="store_true",
        help="Drop old decoded frames when submit is behind. Useful for low-latency streams.",
    )
    parser.add_argument(
        "--keep-all-results",
        action="store_true",
        help=(
            "Do not drop completed outputs before postprocess. This processes every "
            "frame but can reduce NPU utilization if CPU postprocess is slower."
        ),
    )
    parser.add_argument(
        "--preview-all-results",
        action="store_true",
        help=(
            "Preview every postprocessed result in frame order instead of showing "
            "only the newest result. This avoids preview-side drops but can increase "
            "latency and reduce NPU throughput if display is slower than inference."
        ),
    )
    parser.add_argument(
        "--accurate-score-resize",
        action="store_true",
        help=(
            "Resize all class scores before argmax, matching demo_dxnn.py exactly. "
            "Default async video mode uses faster argmax-then-mask-resize."
        ),
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable OpenCV imshow for measuring pipeline throughput.",
    )
    parser.add_argument(
        "-s",
        "--save-video",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Save video output and disable preview. If PATH is omitted, writes "
            "<input>_dxnn_async.mp4 next to the input video."
        ),
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
        default="PIDNet DXNN Async Segmentation",
        help="OpenCV imshow window name.",
    )
    return parser.parse_args()


def validate_positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value


def preview_delay_ms(args: argparse.Namespace) -> int:
    requested = args.video_delay_ms if args.video_delay_ms is not None else args.preview_delay_ms
    return max(1, requested)


def make_inference_option(args: argparse.Namespace) -> InferenceOption:
    option = InferenceOption()
    option.bound_option = BOUND_OPTIONS[args.bound_option]
    if args.buffer_count > 0:
        option.buffer_count = args.buffer_count
    if args.devices is not None:
        option.devices = args.devices
    if args.disable_ort:
        option.use_ort = False
    return option


def tensor_dtype_from_info(info: dict[str, Any]) -> np.dtype:
    return np.dtype(info["dtype"])


def output_to_prediction_fast(
    output: np.ndarray,
    target_hw: tuple[int, int],
    num_classes: int,
) -> np.ndarray:
    if output.ndim == 4 and output.shape[1] == num_classes:
        prediction = np.argmax(output[0], axis=0).astype(np.uint8)
    elif output.ndim == 4 and output.shape[-1] == num_classes:
        prediction = np.argmax(output[0], axis=-1).astype(np.uint8)
    elif output.ndim == 3 and output.shape[0] == num_classes:
        prediction = np.argmax(output, axis=0).astype(np.uint8)
    elif output.ndim == 3 and output.shape[-1] == num_classes:
        prediction = np.argmax(output, axis=-1).astype(np.uint8)
    else:
        prediction = np.squeeze(output)
        if prediction.ndim != 2:
            raise ValueError(f"Unsupported DXNN output shape: {output.shape}")
        prediction = prediction.astype(np.uint8)

    if prediction.shape != target_hw:
        prediction = cv2.resize(
            prediction,
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return prediction.astype(np.uint8, copy=False)


class PIDNetDxnnAsyncDemo:
    def __init__(self, args: argparse.Namespace) -> None:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"DXNN model not found: {model_path}")

        self.option = make_inference_option(args)
        self.engine = InferenceEngine(str(model_path), self.option)
        input_info = self.engine.get_input_tensors_info()
        output_info = self.engine.get_output_tensors_info()
        if not input_info:
            raise RuntimeError("DXNN model has no input tensor metadata.")

        input_shape = input_info[0]["shape"]
        self.input_dtype = tensor_dtype_from_info(input_info[0])
        self.input_layout = infer_input_layout(input_shape, args.input_layout)
        self.input_size = (
            tuple(args.input_size)
            if args.input_size
            else infer_static_input_size(input_shape, self.input_layout)
        )
        self.input_color = args.input_color
        self.expected_input_bytes = self.engine.get_input_size()
        self.output_info = output_info
        self.output_index = args.output_index
        self.num_classes = args.num_classes
        self.accurate_score_resize = args.accurate_score_resize

        print(f"Model: {model_path}")
        print(
            "Input: "
            f"{input_info[0]['name']}, shape={input_shape}, dtype={dtype_name(self.input_dtype)}, "
            f"layout={self.input_layout}, demo_size={self.input_size}, color={self.input_color}"
        )
        if output_info:
            selected_info = output_info[self.output_index]
            print(
                "Output: "
                f"{selected_info['name']}, shape={selected_info['shape']}, "
                f"dtype={dtype_name(selected_info['dtype'])}"
            )
        print(
            "DXRT option: "
            f"bound={self.option.bound_option.name}, devices={self.option.devices}, "
            f"buffer_count={self.option.buffer_count}, use_ort={self.option.use_ort}"
        )

    def close(self) -> None:
        self.engine.dispose()

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        dxnn_input = preprocess_dxnn(
            image_bgr,
            self.input_size,
            self.input_layout,
            self.input_dtype,
            self.input_color,
        )
        if dxnn_input.nbytes != self.expected_input_bytes:
            raise ValueError(
                "DXNN input byte size mismatch: "
                f"got {dxnn_input.nbytes}, expected {self.expected_input_bytes}."
            )
        return dxnn_input

    def submit(self, tensor: np.ndarray) -> int:
        return self.engine.run_async([tensor])

    def wait(self, req_id: int) -> list[np.ndarray]:
        return self.engine.wait(req_id)

    def output_to_prediction(
        self,
        outputs: list[np.ndarray],
        target_hw: tuple[int, int],
    ) -> np.ndarray:
        selected = outputs[self.output_index]
        selected_info = self.output_info[self.output_index] if self.output_info else None
        selected = reshape_output_if_needed(selected, selected_info)
        if self.accurate_score_resize:
            return output_to_prediction(selected, target_hw, self.num_classes)
        return output_to_prediction_fast(selected, target_hw, self.num_classes)


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


def acquire_slot(slots: threading.BoundedSemaphore, stop_event: threading.Event) -> bool:
    while not stop_event.is_set():
        if slots.acquire(timeout=0.05):
            return True
    return False


def read_video_frames(
    video_path: Path,
    frame_queue: queue.Queue[Any],
    errors: queue.Queue[BaseException],
    stats: AsyncStats,
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
            item = CapturedFrame(frame_index, frame, time.perf_counter())
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
        if stop_event.is_set():
            force_put_latest(frame_queue, STOP)
        else:
            put_with_backpressure(frame_queue, STOP, stop_event)


def preprocess_and_submit_worker(
    demo: PIDNetDxnnAsyncDemo,
    frame_queue: queue.Queue[Any],
    request_queue: queue.Queue[Any],
    errors: queue.Queue[BaseException],
    stats: AsyncStats,
    slots: threading.BoundedSemaphore,
    stop_event: threading.Event,
) -> None:
    try:
        while not stop_event.is_set():
            try:
                item = frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item is STOP:
                break
            if not isinstance(item, CapturedFrame):
                continue
            if not acquire_slot(slots, stop_event):
                break

            try:
                preprocess_start = time.perf_counter()
                input_tensor = demo.preprocess(item.image_bgr)
                preprocess_elapsed = time.perf_counter() - preprocess_start
                req_id = demo.submit(input_tensor)
                submitted_at = time.perf_counter()
                stats.add_submitted(preprocess_elapsed, submitted_at)
            except BaseException:
                slots.release()
                raise

            job = SubmittedJob(
                frame_index=item.index,
                image_bgr=item.image_bgr,
                input_tensor=input_tensor,
                req_id=req_id,
                captured_at=item.captured_at,
                submitted_at=submitted_at,
            )
            if not put_with_backpressure(request_queue, job, stop_event):
                break
    except BaseException as exc:
        errors.put(exc)
        stop_event.set()
    finally:
        if stop_event.is_set():
            force_put_latest(request_queue, STOP)
        else:
            put_with_backpressure(request_queue, STOP, stop_event)


def wait_worker(
    demo: PIDNetDxnnAsyncDemo,
    request_queue: queue.Queue[Any],
    output_queue: queue.Queue[Any],
    errors: queue.Queue[BaseException],
    stats: AsyncStats,
    slots: threading.BoundedSemaphore,
    stop_event: threading.Event,
    postprocess_workers: int,
    keep_all_results: bool,
) -> None:
    try:
        while not stop_event.is_set():
            try:
                item = request_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item is STOP:
                break
            if not isinstance(item, SubmittedJob):
                continue

            try:
                outputs = demo.wait(item.req_id)
            finally:
                slots.release()
            completed_at = time.perf_counter()
            stats.add_completed(completed_at - item.submitted_at, completed_at)

            completed = CompletedJob(
                frame_index=item.frame_index,
                image_bgr=item.image_bgr,
                outputs=outputs,
                captured_at=item.captured_at,
                submitted_at=item.submitted_at,
                completed_at=completed_at,
            )
            if keep_all_results:
                if not put_with_backpressure(output_queue, completed, stop_event):
                    break
            elif not put_latest(
                output_queue,
                completed,
                stop_event,
                on_drop=stats.add_dropped_outputs,
            ):
                break
    except BaseException as exc:
        errors.put(exc)
        stop_event.set()
    finally:
        for _ in range(postprocess_workers):
            if stop_event.is_set() or not keep_all_results:
                force_put_latest(output_queue, STOP)
            else:
                put_with_backpressure(output_queue, STOP, stop_event)


def postprocess_worker(
    worker_id: int,
    demo: PIDNetDxnnAsyncDemo,
    output_queue: queue.Queue[Any],
    result_queue: queue.Queue[Any],
    errors: queue.Queue[BaseException],
    stats: AsyncStats,
    stop_event: threading.Event,
    keep_all_results: bool,
) -> None:
    del worker_id
    try:
        while not stop_event.is_set():
            try:
                item = output_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item is STOP:
                break
            if not isinstance(item, CompletedJob):
                continue

            start = time.perf_counter()
            prediction = demo.output_to_prediction(item.outputs, item.image_bgr.shape[:2])
            postprocessed_at = time.perf_counter()
            postprocess_elapsed = postprocessed_at - start
            stats.add_postprocessed(postprocess_elapsed)

            result = PredictionResult(
                frame_index=item.frame_index,
                image_bgr=item.image_bgr,
                prediction=prediction,
                captured_at=item.captured_at,
                submitted_at=item.submitted_at,
                completed_at=item.completed_at,
                postprocessed_at=postprocessed_at,
                async_latency_ms=(item.completed_at - item.submitted_at) * 1000.0,
                postprocess_ms=postprocess_elapsed * 1000.0,
            )
            if keep_all_results:
                if not put_with_backpressure(result_queue, result, stop_event):
                    break
            elif not put_latest(
                result_queue,
                result,
                stop_event,
                on_drop=stats.add_dropped_results,
            ):
                break
    except BaseException as exc:
        errors.put(exc)
        stop_event.set()
    finally:
        if stop_event.is_set() or not keep_all_results:
            force_put_latest(result_queue, STOP)
        else:
            put_with_backpressure(result_queue, STOP, stop_event)


def raise_thread_error(errors: queue.Queue[BaseException]) -> None:
    try:
        error = errors.get_nowait()
    except queue.Empty:
        return
    raise error


def drain_latest_result(
    result_queue: queue.Queue[Any],
    last_displayed_frame: int,
) -> tuple[PredictionResult | None, int, int]:
    latest: PredictionResult | None = None
    drained_results = 0
    stop_count = 0

    while True:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            break

        if result is STOP:
            stop_count += 1
            continue
        if not isinstance(result, PredictionResult):
            continue

        drained_results += 1
        if result.frame_index <= last_displayed_frame:
            continue
        if latest is None or result.frame_index > latest.frame_index:
            latest = result

    if latest is None:
        return None, drained_results, stop_count
    return latest, drained_results - 1, stop_count


def drain_results(result_queue: queue.Queue[Any]) -> tuple[list[PredictionResult], int]:
    results: list[PredictionResult] = []
    stop_count = 0

    while True:
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            break

        if result is STOP:
            stop_count += 1
            continue
        if isinstance(result, PredictionResult):
            results.append(result)

    return results, stop_count


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
    result: PredictionResult,
    stats: AsyncStats,
) -> tuple[float, float, float, float, int]:
    snapshot = stats.snapshot()
    elapsed = max(time.perf_counter() - snapshot.start_time, 1e-9)
    infer_window = (
        snapshot.last_complete_at - snapshot.first_submit_at
        if snapshot.first_submit_at is not None and snapshot.last_complete_at is not None
        else elapsed
    )
    infer_window = max(infer_window, 1e-9)
    npu_fps = snapshot.completed / infer_window
    pipeline_fps = snapshot.postprocessed / elapsed
    display_fps = snapshot.displayed / elapsed
    total_latency_ms = (time.perf_counter() - result.captured_at) * 1000.0
    dropped = (
        snapshot.dropped_inputs
        + snapshot.dropped_outputs
        + snapshot.dropped_results
        + snapshot.skipped_results
    )
    return npu_fps, pipeline_fps, display_fps, total_latency_ms, dropped


def overlay_lines_for_result(
    result: PredictionResult,
    stats: AsyncStats,
    output_label: str = "Preview",
) -> list[str]:
    npu_fps, pipeline_fps, display_fps, total_latency_ms, dropped = metrics_for_result(
        result,
        stats,
    )
    return [
        f"NPU {npu_fps:.1f} FPS",
        f"Pipeline {pipeline_fps:.1f} FPS",
        f"{output_label} {display_fps:.1f} FPS",
        f"Latency {total_latency_ms:.0f} ms",
        f"Dropped {dropped}",
    ]


def title_for_result(
    window_name: str,
    video_path: Path,
    result: PredictionResult,
    stats: AsyncStats,
) -> str:
    snapshot = stats.snapshot()
    npu_fps, _, display_fps, total_latency_ms, dropped = metrics_for_result(
        result,
        stats,
    )
    return (
        f"{window_name} - {video_path.name} - frame {result.frame_index} - "
        f"{npu_fps:.1f} NPU FPS / {display_fps:.1f} preview FPS - "
        f"{result.async_latency_ms:.1f} ms async - {result.postprocess_ms:.1f} ms post - "
        f"{total_latency_ms:.1f} ms latency - inflight {snapshot.inflight_current}/"
        f"{snapshot.inflight_max} - dropped {dropped}"
    )


def print_final_stats(stats: AsyncStats, output_metric_name: str = "preview") -> None:
    snapshot = stats.snapshot()
    elapsed = max(time.perf_counter() - snapshot.start_time, 1e-9)
    infer_window = (
        snapshot.last_complete_at - snapshot.first_submit_at
        if snapshot.first_submit_at is not None and snapshot.last_complete_at is not None
        else elapsed
    )
    infer_window = max(infer_window, 1e-9)
    npu_fps = snapshot.completed / infer_window
    pipeline_fps = snapshot.postprocessed / elapsed
    display_fps = snapshot.displayed / elapsed
    avg_pre_ms = (
        snapshot.sum_preprocess / snapshot.submitted * 1000.0
        if snapshot.submitted
        else 0.0
    )
    avg_async_ms = (
        snapshot.sum_async_latency / snapshot.completed * 1000.0
        if snapshot.completed
        else 0.0
    )
    avg_post_ms = (
        snapshot.sum_postprocess / snapshot.postprocessed * 1000.0
        if snapshot.postprocessed
        else 0.0
    )
    print(
        "Async DXNN video summary: "
        f"captured={snapshot.captured}, submitted={snapshot.submitted}, "
        f"completed={snapshot.completed}, postprocessed={snapshot.postprocessed}, "
        f"displayed={snapshot.displayed}, npu_fps={npu_fps:.2f}, "
        f"pipeline_fps={pipeline_fps:.2f}, {output_metric_name}_fps={display_fps:.2f}, "
        f"avg_pre_ms={avg_pre_ms:.2f}, avg_async_ms={avg_async_ms:.2f}, "
        f"avg_post_ms={avg_post_ms:.2f}, inflight_max={snapshot.inflight_max}, "
        f"dropped_input={snapshot.dropped_inputs}, "
        f"dropped_output={snapshot.dropped_outputs}, "
        f"dropped_result={snapshot.dropped_results}, "
        f"skipped_preview={snapshot.skipped_results}"
    )


def resolve_save_video_path(video_path: Path, save_arg: str | None) -> Path | None:
    if save_arg is None:
        return None
    if save_arg == "":
        return video_path.with_name(f"{video_path.stem}_dxnn_async.mp4")

    output_path = Path(save_arg)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".mp4")
    return output_path


def video_source_fps(video_path: Path) -> float:
    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            return 30.0
        fps = video.get(cv2.CAP_PROP_FPS)
    finally:
        video.release()

    if fps <= 0 or not np.isfinite(fps):
        return 30.0
    return float(fps)


def video_fourcc_for_path(output_path: Path) -> int:
    if output_path.suffix.lower() == ".avi":
        return cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter_fourcc(*"mp4v")


def open_video_writer(
    output_path: Path,
    first_frame: np.ndarray,
    fps: float,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_size = (first_frame.shape[1], first_frame.shape[0])
    writer = cv2.VideoWriter(
        str(output_path),
        video_fourcc_for_path(output_path),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    return writer


def make_output_frame(
    result: PredictionResult,
    args: argparse.Namespace,
    stats: AsyncStats,
    output_label: str,
    max_display_size: int | None = None,
) -> np.ndarray:
    frame = make_visualization(
        result.image_bgr,
        result.prediction,
        args.view,
        args.alpha,
    )
    if max_display_size is not None:
        frame = resize_for_display(frame, max_display_size)
    return add_status_overlay_top_right(
        frame,
        overlay_lines_for_result(result, stats, output_label),
    )


def run_video_async(demo: PIDNetDxnnAsyncDemo, video_path: Path, args: argparse.Namespace) -> None:
    save_video_path = resolve_save_video_path(video_path, args.save_video)
    save_video = save_video_path is not None
    if save_video and save_video_path.resolve() == video_path.resolve():
        raise ValueError("Output video path must be different from the input video path.")

    max_inflight = args.max_inflight if args.max_inflight > 0 else demo.option.buffer_count
    max_inflight = validate_positive(max_inflight, "--max-inflight")
    frame_queue_size = (
        args.frame_queue_size
        if args.frame_queue_size > 0
        else max(2 * max_inflight, 8)
    )
    result_queue_size = validate_positive(args.result_queue_size, "--result-queue-size")
    postprocess_workers = validate_positive(args.postprocess_workers, "--postprocess-workers")
    output_queue_size = args.output_queue_size if args.output_queue_size > 0 else max_inflight
    output_queue_size = max(output_queue_size, postprocess_workers + 1)
    result_queue_size = max(result_queue_size, postprocess_workers + 1)
    delay_ms = preview_delay_ms(args)
    preview_enabled = not args.no_preview and not save_video
    preview_all_results = args.preview_all_results and preview_enabled
    ordered_output = save_video or preview_all_results
    drop_input_frames = args.drop_input_frames and not ordered_output
    keep_all_results = args.keep_all_results or ordered_output

    frame_queue: queue.Queue[Any] = queue.Queue(maxsize=frame_queue_size)
    request_queue: queue.Queue[Any] = queue.Queue(maxsize=max_inflight)
    output_queue: queue.Queue[Any] = queue.Queue(maxsize=output_queue_size)
    result_queue: queue.Queue[Any] = queue.Queue(maxsize=result_queue_size)
    errors: queue.Queue[BaseException] = queue.Queue()
    stats = AsyncStats()
    stop_event = threading.Event()
    slots = threading.BoundedSemaphore(max_inflight)

    threads = [
        threading.Thread(
            target=read_video_frames,
            name="video-reader",
            args=(
                video_path,
                frame_queue,
                errors,
                stats,
                stop_event,
                drop_input_frames,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=preprocess_and_submit_worker,
            name="dxnn-submit",
            args=(
                demo,
                frame_queue,
                request_queue,
                errors,
                stats,
                slots,
                stop_event,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=wait_worker,
            name="dxnn-wait",
            args=(
                demo,
                request_queue,
                output_queue,
                errors,
                stats,
                slots,
                stop_event,
                postprocess_workers,
                keep_all_results,
            ),
            daemon=True,
        ),
    ]
    threads.extend(
        threading.Thread(
            target=postprocess_worker,
            name=f"postprocess-{idx + 1}",
            args=(
                idx,
                demo,
                output_queue,
                result_queue,
                errors,
                stats,
                stop_event,
                keep_all_results,
            ),
            daemon=True,
        )
        for idx in range(postprocess_workers)
    )

    print(
        "Async DXNN pipeline: "
        f"max_inflight={max_inflight}, frame_queue={frame_queue_size}, "
        f"output_queue={output_queue_size}, result_queue={result_queue_size}, "
        f"postprocess_workers={postprocess_workers}, preview_delay_ms={delay_ms}, "
        f"drop_input_frames={drop_input_frames}, "
        f"keep_all_results={keep_all_results}, "
        f"preview_all_results={preview_all_results}, "
        f"accurate_score_resize={args.accurate_score_resize}, "
        f"preview={preview_enabled}, "
        f"save_video={save_video_path}"
    )
    if save_video:
        print(f"Saving video to: {save_video_path}")
        print("Preview is disabled while saving. Press Ctrl-C to quit.")
    elif not preview_enabled:
        print("Press Ctrl-C to quit.")
    else:
        print("Press q or Esc in the display window to quit.")

    for thread in threads:
        thread.start()

    last_displayed_frame = 0
    result_stop_count = 0
    pending_ordered_results: dict[int, PredictionResult] = {}
    next_ordered_frame = 1
    saved_frames = 0
    writer: cv2.VideoWriter | None = None
    save_fps = video_source_fps(video_path) if save_video else 0.0
    try:
        while True:
            raise_thread_error(errors)
            if ordered_output:
                results, stop_count = drain_results(result_queue)
                result_stop_count += stop_count
                for result in results:
                    pending_ordered_results[result.frame_index] = result

                key = -1
                emitted = False
                while next_ordered_frame in pending_ordered_results:
                    result = pending_ordered_results.pop(next_ordered_frame)
                    last_displayed_frame = result.frame_index
                    stats.add_displayed()
                    frame = make_output_frame(
                        result,
                        args,
                        stats,
                        "Saved" if save_video else "Preview",
                        None if save_video else args.max_display_size,
                    )
                    if save_video and writer is None:
                        writer = open_video_writer(save_video_path, frame, save_fps)
                    if save_video:
                        writer.write(frame)
                        saved_frames += 1
                        key = -1
                    else:
                        key = show_async_result(
                            args.window_name,
                            title_for_result(args.window_name, video_path, result, stats),
                            frame,
                            delay_ms,
                        )
                    next_ordered_frame += 1
                    emitted = True
                    if key in QUIT_KEYS:
                        break

                if key in QUIT_KEYS:
                    break
                if not emitted:
                    if save_video:
                        time.sleep(0.001)
                        key = -1
                    else:
                        key = wait_for_preview_key(delay_ms)
            else:
                latest, skipped, stop_count = drain_latest_result(
                    result_queue,
                    last_displayed_frame,
                )
                result_stop_count += stop_count
                stats.add_skipped_results(skipped)

                if latest is not None:
                    last_displayed_frame = latest.frame_index
                    stats.add_displayed()
                    if not preview_enabled:
                        key = -1
                    else:
                        display = make_output_frame(
                            latest,
                            args,
                            stats,
                            "Preview",
                            args.max_display_size,
                        )
                        key = show_async_result(
                            args.window_name,
                            title_for_result(args.window_name, video_path, latest, stats),
                            display,
                            delay_ms,
                        )
                else:
                    if not preview_enabled:
                        time.sleep(0.001)
                        key = -1
                    else:
                        key = wait_for_preview_key(delay_ms)

            if key in QUIT_KEYS:
                break
            if result_stop_count >= postprocess_workers and result_queue.empty():
                if ordered_output and pending_ordered_results:
                    for frame_index in sorted(pending_ordered_results):
                        result = pending_ordered_results[frame_index]
                        last_displayed_frame = result.frame_index
                        stats.add_displayed()
                        frame = make_output_frame(
                            result,
                            args,
                            stats,
                            "Saved" if save_video else "Preview",
                            None if save_video else args.max_display_size,
                        )
                        if save_video and writer is None:
                            writer = open_video_writer(save_video_path, frame, save_fps)
                        if save_video:
                            writer.write(frame)
                            saved_frames += 1
                        else:
                            key = show_async_result(
                                args.window_name,
                                title_for_result(args.window_name, video_path, result, stats),
                                frame,
                                delay_ms,
                            )
                            if key in QUIT_KEYS:
                                break
                    pending_ordered_results.clear()
                break
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()
        if writer is not None:
            writer.release()
        if save_video:
            print(f"Saved video frames: {saved_frames} -> {save_video_path}")
        raise_thread_error(errors)
        print_final_stats(stats, "saved" if save_video else "preview")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    wants_save_video = args.save_video is not None
    if wants_save_video:
        args.no_preview = True

    if input_path.is_dir() or is_image_file(input_path):
        if wants_save_video:
            raise ValueError("-s/--save-video is only supported for video input.")
        demo = PIDNetDxnnDemo(args)
        try:
            create_window(args.window_name)
            if input_path.is_dir():
                run_image_folder(demo, input_path, args)
            else:
                run_image(demo, input_path, args, wait_ms=0)
        finally:
            demo.close()
            destroy_windows()
        return

    if not is_video_file(input_path):
        raise ValueError(
            f"Unsupported input type: {input_path}. "
            "Use an image file, image directory, or video file."
        )

    demo = PIDNetDxnnAsyncDemo(args)
    try:
        if not args.no_preview:
            create_window(args.window_name)
        run_video_async(demo, input_path, args)
    finally:
        demo.close()
        destroy_windows()


if __name__ == "__main__":
    main()
