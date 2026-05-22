#!/usr/bin/env python3
"""Minimal async PIDNet DXNN video demo."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dx_engine import InferenceEngine, InferenceOption


QUIT_KEYS = {27, ord("q"), ord("Q")}
STOP = object()
NUM_CLASSES = 19
ALPHA = 0.55
POSTPROCESS_WORKERS = 4

CITYSCAPES_COLORS_BGR = np.array(
    [
        (128, 64, 128),
        (232, 35, 244),
        (70, 70, 70),
        (156, 102, 102),
        (153, 153, 190),
        (153, 153, 153),
        (30, 170, 250),
        (0, 220, 220),
        (35, 142, 107),
        (152, 251, 152),
        (180, 130, 70),
        (60, 20, 220),
        (0, 0, 255),
        (142, 0, 0),
        (70, 0, 0),
        (100, 60, 0),
        (100, 80, 0),
        (230, 0, 0),
        (32, 11, 119),
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class Frame:
    index: int
    image: np.ndarray
    captured_at: float


@dataclass(frozen=True)
class Submitted:
    index: int
    image: np.ndarray
    req_id: int
    captured_at: float
    submitted_at: float


@dataclass(frozen=True)
class Completed:
    index: int
    image: np.ndarray
    outputs: list[np.ndarray]
    captured_at: float
    submitted_at: float
    completed_at: float


@dataclass(frozen=True)
class Result:
    index: int
    image: np.ndarray
    prediction: np.ndarray
    captured_at: float
    submitted_at: float
    completed_at: float
    post_ms: float


class Stats:
    def __init__(self) -> None:
        self.start_at = time.perf_counter()
        self.captured = 0
        self.submitted = 0
        self.completed = 0
        self.postprocessed = 0
        self.displayed = 0
        self.first_submit_at: float | None = None
        self.last_complete_at: float | None = None
        self.sum_pre_ms = 0.0
        self.sum_async_ms = 0.0
        self.sum_post_ms = 0.0
        self.inflight = 0
        self.inflight_max = 0
        self.lock = threading.Lock()

    def add_captured(self) -> None:
        with self.lock:
            self.captured += 1

    def add_submitted(self, pre_ms: float, submitted_at: float) -> None:
        with self.lock:
            self.submitted += 1
            self.sum_pre_ms += pre_ms
            self.inflight += 1
            self.inflight_max = max(self.inflight_max, self.inflight)
            if self.first_submit_at is None:
                self.first_submit_at = submitted_at

    def add_completed(self, async_ms: float, completed_at: float) -> None:
        with self.lock:
            self.completed += 1
            self.sum_async_ms += async_ms
            self.last_complete_at = completed_at
            self.inflight = max(0, self.inflight - 1)

    def add_postprocessed(self, post_ms: float) -> None:
        with self.lock:
            self.postprocessed += 1
            self.sum_post_ms += post_ms

    def add_displayed(self) -> None:
        with self.lock:
            self.displayed += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "start_at": self.start_at,
                "captured": self.captured,
                "submitted": self.submitted,
                "completed": self.completed,
                "postprocessed": self.postprocessed,
                "displayed": self.displayed,
                "first_submit_at": self.first_submit_at,
                "last_complete_at": self.last_complete_at,
                "sum_pre_ms": self.sum_pre_ms,
                "sum_async_ms": self.sum_async_ms,
                "sum_post_ms": self.sum_post_ms,
                "inflight": self.inflight,
                "inflight_max": self.inflight_max,
            }


class PIDNet:
    def __init__(self, model_path: Path) -> None:
        option = InferenceOption()
        option.bound_option = InferenceOption.BOUND_OPTION.NPU_ALL
        self.engine = InferenceEngine(str(model_path), option)
        self.option = option

        input_info = self.engine.get_input_tensors_info()[0]
        output_info = self.engine.get_output_tensors_info()
        self.input_shape = input_info["shape"]
        self.input_dtype = np.dtype(input_info["dtype"])
        self.input_layout = input_layout(self.input_shape)
        self.input_size = static_input_size(self.input_shape, self.input_layout)
        self.input_bytes = self.engine.get_input_size()
        self.output_info = output_info
        self.output_index = -1

        selected = output_info[self.output_index]
        print(f"Model: {model_path}")
        print(
            "Input: "
            f"{input_info['name']}, shape={self.input_shape}, "
            f"dtype={self.input_dtype.name}, layout={self.input_layout}, "
            f"demo_size={self.input_size}, color=rgb"
        )
        print(
            "Output: "
            f"{selected['name']}, shape={selected['shape']}, "
            f"dtype={np.dtype(selected['dtype']).name}"
        )
        print(
            "DXRT option: "
            f"bound={self.option.bound_option.name}, "
            f"buffer_count={self.option.buffer_count}, use_ort={self.option.use_ort}"
        )

    def close(self) -> None:
        self.engine.dispose()

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = self.input_size
        resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if np.issubdtype(self.input_dtype, np.floating):
            tensor = resized.astype(np.float32) / 255.0
            tensor -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
            tensor /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        else:
            tensor = resized.astype(self.input_dtype, copy=False)

        if self.input_layout == "nchw":
            tensor = tensor.transpose(2, 0, 1)

        tensor = np.ascontiguousarray(np.expand_dims(tensor, 0), dtype=self.input_dtype)
        if tensor.nbytes != self.input_bytes:
            raise ValueError(
                f"Input byte size mismatch: got {tensor.nbytes}, expected {self.input_bytes}"
            )
        return tensor

    def submit(self, image_bgr: np.ndarray) -> int:
        return self.engine.run_async([self.preprocess(image_bgr)])

    def wait(self, req_id: int) -> list[np.ndarray]:
        return self.engine.wait(req_id)

    def prediction(self, outputs: list[np.ndarray], target_hw: tuple[int, int]) -> np.ndarray:
        output = outputs[self.output_index]
        output = reshape_output(output, self.output_info[self.output_index])
        return output_to_prediction(output, target_hw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PIDNet DXNN async video demo.")
    parser.add_argument("-m", "--model", required=True, help="DXNN model path.")
    parser.add_argument("-v", "--video", required=True, help="Input video path.")
    return parser.parse_args()


def input_layout(shape: list[int]) -> str:
    if len(shape) != 4:
        raise ValueError(f"Unsupported input shape: {shape}")
    if shape[-1] in (1, 3, 4):
        return "nhwc"
    if shape[1] in (1, 3, 4):
        return "nchw"
    raise ValueError(f"Could not infer input layout from shape: {shape}")


def static_input_size(shape: list[int], layout: str) -> tuple[int, int]:
    height, width = (shape[1], shape[2]) if layout == "nhwc" else (shape[2], shape[3])
    if height <= 0 or width <= 0:
        raise ValueError(f"Input shape is not static: {shape}")
    return height, width


def reshape_output(output: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    shape = info.get("shape")
    if output.ndim == 1 and shape and output.size == int(np.prod(shape)):
        return output.reshape(tuple(shape))
    return output


def resize_scores(scores: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if scores.shape[-2:] == target_hw:
        return scores

    target_h, target_w = target_hw
    resized = np.empty((scores.shape[0], target_h, target_w), dtype=np.float32)
    for class_idx in range(scores.shape[0]):
        resized[class_idx] = cv2.resize(
            scores[class_idx],
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
    return resized


def output_to_prediction(output: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if output.ndim == 4 and output.shape[1] == NUM_CLASSES:
        scores = resize_scores(output[0], target_hw)
        return np.argmax(scores, axis=0).astype(np.uint8)
    if output.ndim == 4 and output.shape[-1] == NUM_CLASSES:
        scores = resize_scores(output[0].transpose(2, 0, 1), target_hw)
        return np.argmax(scores, axis=0).astype(np.uint8)
    if output.ndim == 3 and output.shape[0] == NUM_CLASSES:
        scores = resize_scores(output, target_hw)
        return np.argmax(scores, axis=0).astype(np.uint8)
    if output.ndim == 3 and output.shape[-1] == NUM_CLASSES:
        scores = resize_scores(output.transpose(2, 0, 1), target_hw)
        return np.argmax(scores, axis=0).astype(np.uint8)

    prediction = np.squeeze(output)
    if prediction.ndim != 2:
        raise ValueError(f"Unsupported output shape: {output.shape}")
    if prediction.shape != target_hw:
        prediction = cv2.resize(
            prediction.astype(np.uint8),
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return prediction.astype(np.uint8)


def put_blocking(items: queue.Queue[Any], item: Any, stop_event: threading.Event) -> bool:
    while not stop_event.is_set():
        try:
            items.put(item, timeout=0.05)
            return True
        except queue.Full:
            continue
    return False


def force_put(items: queue.Queue[Any], item: Any) -> None:
    while True:
        try:
            items.put_nowait(item)
            return
        except queue.Full:
            try:
                items.get_nowait()
            except queue.Empty:
                pass


def acquire(slots: threading.BoundedSemaphore, stop_event: threading.Event) -> bool:
    while not stop_event.is_set():
        if slots.acquire(timeout=0.05):
            return True
    return False


def reader(video_path: Path, frames: queue.Queue[Any], errors: queue.Queue[BaseException], stats: Stats, stop_event: threading.Event) -> None:
    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        index = 0
        while not stop_event.is_set():
            ok, image = video.read()
            if not ok:
                break
            index += 1
            stats.add_captured()
            if not put_blocking(frames, Frame(index, image, time.perf_counter()), stop_event):
                break
    except BaseException as exc:
        errors.put(exc)
        stop_event.set()
    finally:
        video.release()
        put_blocking(frames, STOP, stop_event) if not stop_event.is_set() else force_put(frames, STOP)


def submitter(model: PIDNet, frames: queue.Queue[Any], requests: queue.Queue[Any], errors: queue.Queue[BaseException], stats: Stats, slots: threading.BoundedSemaphore, stop_event: threading.Event) -> None:
    try:
        while not stop_event.is_set():
            try:
                item = frames.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is STOP:
                break
            if not acquire(slots, stop_event):
                break
            try:
                start = time.perf_counter()
                req_id = model.submit(item.image)
                submitted_at = time.perf_counter()
                stats.add_submitted((submitted_at - start) * 1000.0, submitted_at)
            except BaseException:
                slots.release()
                raise
            job = Submitted(item.index, item.image, req_id, item.captured_at, submitted_at)
            if not put_blocking(requests, job, stop_event):
                break
    except BaseException as exc:
        errors.put(exc)
        stop_event.set()
    finally:
        put_blocking(requests, STOP, stop_event) if not stop_event.is_set() else force_put(requests, STOP)


def waiter(model: PIDNet, requests: queue.Queue[Any], completed: queue.Queue[Any], errors: queue.Queue[BaseException], stats: Stats, slots: threading.BoundedSemaphore, stop_event: threading.Event) -> None:
    try:
        while not stop_event.is_set():
            try:
                item = requests.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is STOP:
                break
            try:
                outputs = model.wait(item.req_id)
            finally:
                slots.release()
            completed_at = time.perf_counter()
            stats.add_completed((completed_at - item.submitted_at) * 1000.0, completed_at)
            done = Completed(
                item.index,
                item.image,
                outputs,
                item.captured_at,
                item.submitted_at,
                completed_at,
            )
            if not put_blocking(completed, done, stop_event):
                break
    except BaseException as exc:
        errors.put(exc)
        stop_event.set()
    finally:
        for _ in range(POSTPROCESS_WORKERS):
            put_blocking(completed, STOP, stop_event) if not stop_event.is_set() else force_put(completed, STOP)


def postprocessor(model: PIDNet, completed: queue.Queue[Any], results: queue.Queue[Any], errors: queue.Queue[BaseException], stats: Stats, stop_event: threading.Event) -> None:
    try:
        while not stop_event.is_set():
            try:
                item = completed.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is STOP:
                break

            start = time.perf_counter()
            prediction = model.prediction(item.outputs, item.image.shape[:2])
            post_ms = (time.perf_counter() - start) * 1000.0
            stats.add_postprocessed(post_ms)
            result = Result(
                item.index,
                item.image,
                prediction,
                item.captured_at,
                item.submitted_at,
                item.completed_at,
                post_ms,
            )
            if not put_blocking(results, result, stop_event):
                break
    except BaseException as exc:
        errors.put(exc)
        stop_event.set()
    finally:
        put_blocking(results, STOP, stop_event) if not stop_event.is_set() else force_put(results, STOP)


def raise_thread_error(errors: queue.Queue[BaseException]) -> None:
    try:
        error = errors.get_nowait()
    except queue.Empty:
        return
    raise error


def drain_results(results: queue.Queue[Any]) -> tuple[list[Result], int]:
    drained: list[Result] = []
    stops = 0
    while True:
        try:
            item = results.get_nowait()
        except queue.Empty:
            break
        if item is STOP:
            stops += 1
        else:
            drained.append(item)
    return drained, stops


def colorize(prediction: np.ndarray) -> np.ndarray:
    return CITYSCAPES_COLORS_BGR[np.clip(prediction, 0, NUM_CLASSES - 1)]


def make_frame(result: Result, stats: Stats) -> np.ndarray:
    mask = colorize(result.prediction)
    frame = cv2.addWeighted(result.image, 1.0 - ALPHA, mask, ALPHA, 0.0)
    return add_status_overlay(frame, result, stats)


def add_status_overlay(image: np.ndarray, result: Result, stats: Stats) -> np.ndarray:
    snapshot = stats.snapshot()
    now = time.perf_counter()
    elapsed = max(now - snapshot["start_at"], 1e-9)
    infer_window = elapsed
    if snapshot["first_submit_at"] is not None and snapshot["last_complete_at"] is not None:
        infer_window = max(snapshot["last_complete_at"] - snapshot["first_submit_at"], 1e-9)

    lines = [
        f"NPU {snapshot['completed'] / infer_window:.1f} FPS",
        f"Pipeline {snapshot['postprocessed'] / elapsed:.1f} FPS",
        f"Preview {snapshot['displayed'] / elapsed:.1f} FPS",
        f"Latency {(now - result.captured_at) * 1000.0:.0f} ms",
        "Dropped 0",
    ]

    output = image.copy()
    height, width = output.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.75, width / 1600.0))
    thickness = max(1, int(round(font_scale * 2.0)))
    padding = max(6, int(round(10 * font_scale)))
    margin = max(8, int(round(14 * font_scale)))
    gap = max(4, int(round(7 * font_scale)))
    sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    box_w = max(size[0] for size in sizes) + padding * 2
    box_h = sum(size[1] for size in sizes) + gap * (len(lines) - 1) + padding * 2
    x0 = max(0, width - box_w - margin)
    y0 = margin
    x1 = min(width - 1, width - margin)
    y1 = min(height - 1, y0 + box_h)

    shaded = output.copy()
    cv2.rectangle(shaded, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(shaded, 0.58, output, 0.42, 0.0, output)

    y = y0 + padding
    for idx, line in enumerate(lines):
        y += sizes[idx][1]
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
        y += gap
    return output


def show(window_name: str, video_path: Path, result: Result, display: np.ndarray, stats: Stats) -> int:
    snapshot = stats.snapshot()
    elapsed = max(time.perf_counter() - snapshot["start_at"], 1e-9)
    preview_fps = snapshot["displayed"] / elapsed
    title = (
        f"{window_name} - {video_path.name} - frame {result.index} - "
        f"{preview_fps:.1f} preview FPS - {result.post_ms:.1f} ms post - "
        f"inflight {snapshot['inflight']}/{snapshot['inflight_max']}"
    )
    try:
        cv2.imshow(window_name, display)
        try:
            cv2.setWindowTitle(window_name, title)
        except cv2.error:
            pass
        return cv2.waitKey(1) & 0xFF
    except cv2.error as exc:
        raise RuntimeError(
            "OpenCV GUI is unavailable. Install opencv-python and run with display support."
        ) from exc


def print_summary(stats: Stats) -> None:
    snapshot = stats.snapshot()
    elapsed = max(time.perf_counter() - snapshot["start_at"], 1e-9)
    infer_window = elapsed
    if snapshot["first_submit_at"] is not None and snapshot["last_complete_at"] is not None:
        infer_window = max(snapshot["last_complete_at"] - snapshot["first_submit_at"], 1e-9)

    print(
        "Summary: "
        f"captured={snapshot['captured']}, submitted={snapshot['submitted']}, "
        f"completed={snapshot['completed']}, postprocessed={snapshot['postprocessed']}, "
        f"displayed={snapshot['displayed']}, "
        f"npu_fps={snapshot['completed'] / infer_window:.2f}, "
        f"pipeline_fps={snapshot['postprocessed'] / elapsed:.2f}, "
        f"preview_fps={snapshot['displayed'] / elapsed:.2f}, "
        "dropped=0"
    )


def run(model: PIDNet, video_path: Path) -> None:
    max_inflight = max(1, int(model.option.buffer_count))
    frame_queue: queue.Queue[Any] = queue.Queue(maxsize=max(2 * max_inflight, 8))
    request_queue: queue.Queue[Any] = queue.Queue(maxsize=max_inflight)
    output_queue: queue.Queue[Any] = queue.Queue(maxsize=max(max_inflight, POSTPROCESS_WORKERS + 1))
    result_queue: queue.Queue[Any] = queue.Queue(maxsize=POSTPROCESS_WORKERS + 1)
    errors: queue.Queue[BaseException] = queue.Queue()
    stop_event = threading.Event()
    stats = Stats()
    slots = threading.BoundedSemaphore(max_inflight)
    window_name = "PIDNet DXNN Demo"

    threads = [
        threading.Thread(target=reader, args=(video_path, frame_queue, errors, stats, stop_event), daemon=True),
        threading.Thread(target=submitter, args=(model, frame_queue, request_queue, errors, stats, slots, stop_event), daemon=True),
        threading.Thread(target=waiter, args=(model, request_queue, output_queue, errors, stats, slots, stop_event), daemon=True),
    ]
    threads.extend(
        threading.Thread(target=postprocessor, args=(model, output_queue, result_queue, errors, stats, stop_event), daemon=True)
        for _ in range(POSTPROCESS_WORKERS)
    )

    #try:
    #    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    #except cv2.error as exc:
    #    raise RuntimeError("OpenCV could not create a display window.") from exc

    print(
        "Pipeline: "
        f"max_inflight={max_inflight}, postprocess_workers={POSTPROCESS_WORKERS}, "
        "preview_all_results=True, input_color=rgb"
    )
    print("Press q or Esc in the display window to quit.")

    for thread in threads:
        thread.start()

    pending: dict[int, Result] = {}
    next_frame = 1
    stop_count = 0
    try:
        while True:
            raise_thread_error(errors)
            drained, stops = drain_results(result_queue)
            stop_count += stops
            for result in drained:
                pending[result.index] = result

            key = -1
            emitted = False
            while next_frame in pending:
                result = pending.pop(next_frame)
                stats.add_displayed()
                key = show(window_name, video_path, result, make_frame(result, stats), stats)
                next_frame += 1
                emitted = True
                if key in QUIT_KEYS:
                    break

            if key in QUIT_KEYS:
                break
            if not emitted:
                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS:
                    break
                time.sleep(0.001)
            if stop_count >= POSTPROCESS_WORKERS and result_queue.empty():
                break
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()
        cv2.destroyAllWindows()
        raise_thread_error(errors)
        print_summary(stats)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    video_path = Path(args.video)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = PIDNet(model_path)
    try:
        run(model, video_path)
    finally:
        model.close()


if __name__ == "__main__":
    main()
