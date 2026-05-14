"""
Backend — WebSocket frame receiver → MoveNet → sliding window → LLM context.

Flow:
    Browser camera ──JPEG binary──► WebSocket /stream
                                         │
                                  thread pool executor
                                  (non-blocking inference)
                                         │
                                   MoveNetInference        every frame
                                         │
                                    FrameBuffer            sliding window
                                    (last N seconds)
                                         │  flush every send_interval
                                    ContextBatch ──JSON──► client / LLM service

The ContextBatch contains:
  - keypoint sequence for every frame in the window  (temporal motion context)
  - a few evenly-sampled annotated images            (visual context for LLM)

Frontend (minimal):
    const ws = new WebSocket("ws://localhost:8000/stream")
    ws.onopen  = () => ws.send(JSON.stringify({ exercise: "squat" }))
    ws.onmessage = e => sendToLLM(JSON.parse(e.data))

    function sendFrame() {
        canvas.toBlob(blob => ws.readyState === 1 && ws.send(blob), "image/jpeg", 0.75)
        requestAnimationFrame(sendFrame)
    }
    sendFrame()

Run:
    uvicorn backend:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from movenet import FrameContext, KP, Keypoints, MoveNetInference

# One thread pool shared across all connections for MoveNet inference.
# MoveNet is CPU/GPU bound — running it in a thread keeps the event loop free.
_executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# ContextBatch — one window of frames sent to the LLM
# ---------------------------------------------------------------------------

@dataclass
class ContextBatch:
    """
    A sliding-window snapshot of the last `window_seconds` of motion.

    exercise          — label set by the client
    timestamp         — Unix time this batch was built
    window_seconds    — duration of the captured window
    frame_count       — number of MoveNet frames in the window
    keypoints_sequence — keypoints for every frame [{name: {x,y,conf}}, ...]
    sampled_images    — base64 JPEGs evenly sampled from the window
    batch_index       — monotonically increasing per connection
    """
    exercise:           Optional[str]
    timestamp:          float
    window_seconds:     float
    frame_count:        int
    keypoints_sequence: list[dict[str, dict]]
    sampled_images:     list[str]               # base64 JPEGs
    batch_index:        int

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=lambda x: float(x) if hasattr(x, "__float__") else x)

    def to_anthropic_messages(self) -> list[dict]:
        """
        Format for client.messages.create(messages=...).
        Sends sampled images as vision blocks + keypoint sequence as text.
        """
        content: list[dict] = []

        for img_b64 in self.sampled_images:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
            })

        kp_text = _format_keypoint_sequence(self.keypoints_sequence)
        meta = (
            f"Exercise: {self.exercise or 'unknown'}\n"
            f"Window: {self.window_seconds:.1f}s  |  "
            f"Frames: {self.frame_count}  |  "
            f"Images shown: {len(self.sampled_images)}\n\n"
            f"{kp_text}"
        )
        content.append({"type": "text", "text": meta})

        return [{"role": "user", "content": content}]


def _format_keypoint_sequence(seq: list[dict[str, dict]]) -> str:
    """Compact text representation of keypoint motion across frames."""
    if not seq:
        return ""
    lines = [f"Keypoint motion across {len(seq)} frames:"]
    # Only include body keypoints relevant to form (skip face)
    tracked = [
        "left_shoulder", "right_shoulder",
        "left_elbow",    "right_elbow",
        "left_wrist",    "right_wrist",
        "left_hip",      "right_hip",
        "left_knee",     "right_knee",
        "left_ankle",    "right_ankle",
    ]
    for name in tracked:
        positions = [
            f"({f[name]['x']:.2f},{f[name]['y']:.2f})"
            for f in seq
            if name in f and f[name]["visible"]
        ]
        if positions:
            # Downsample to at most 10 positions for brevity
            step = max(1, len(positions) // 10)
            lines.append(f"  {name}: {' → '.join(positions[::step])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FrameBuffer — sliding window of keypoints + raw frames
# ---------------------------------------------------------------------------

@dataclass
class _Frame:
    keypoints: dict[str, dict]
    raw:       np.ndarray       # BGR, for image sampling
    captured_at: float


class FrameBuffer:
    """
    Accumulates MoveNet output for a rolling time window.
    Thread-safe: add() is called from the executor thread,
    flush() from the async handler.
    """

    def __init__(
        self,
        window_seconds: float = 2.0,
        sampled_images: int   = 4,
        jpeg_quality:   int   = 70,
        conf_threshold: float = 0.3,
    ):
        self.window_seconds = window_seconds
        self.n_images       = sampled_images
        self.jpeg_quality   = jpeg_quality
        self.conf_threshold = conf_threshold
        self._buf: deque[_Frame] = deque()

    def add(self, frame: np.ndarray, kp: Keypoints) -> None:
        now = time.monotonic()
        self._buf.append(_Frame(
            keypoints=kp.to_dict(self.conf_threshold),
            raw=frame,
            captured_at=now,
        ))
        # Evict frames outside the window
        cutoff = now - self.window_seconds
        while self._buf and self._buf[0].captured_at < cutoff:
            self._buf.popleft()

    def build(
        self,
        exercise:    Optional[str],
        batch_index: int,
        model:       MoveNetInference,
    ) -> Optional[ContextBatch]:
        if not self._buf:
            return None

        frames = list(self._buf)
        kp_seq = [f.keypoints for f in frames]

        # Sample N evenly spaced frames for images
        indices = _even_indices(len(frames), self.n_images)
        images  = [
            _encode_jpeg(
                model.draw_skeleton(
                    frames[i].raw,
                    _reconstruct_kp(frames[i].keypoints),
                ),
                self.jpeg_quality,
            )
            for i in indices
        ]

        duration = frames[-1].captured_at - frames[0].captured_at if len(frames) > 1 else 0.0

        return ContextBatch(
            exercise=exercise,
            timestamp=time.time(),
            window_seconds=duration,
            frame_count=len(frames),
            keypoints_sequence=kp_seq,
            sampled_images=images,
            batch_index=batch_index,
        )

    def clear(self) -> None:
        self._buf.clear()


def _even_indices(total: int, n: int) -> list[int]:
    if total <= n:
        return list(range(total))
    step = total / n
    return [int(i * step) for i in range(n)]


def _encode_jpeg(frame: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed.")
    return base64.b64encode(buf).decode("utf-8")


def _reconstruct_kp(kp_dict: dict[str, dict]) -> Keypoints:
    """Reconstruct a Keypoints object from its serialised dict for draw_skeleton."""
    name_to_idx = {v.name.lower(): int(v) for v in KP}
    arr = np.zeros((17, 3), dtype=np.float32)
    for name, v in kp_dict.items():
        idx = name_to_idx.get(name)
        if idx is not None:
            arr[idx] = [v["y"], v["x"], v["confidence"]]
    return Keypoints(data=arr)


def _decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


# ---------------------------------------------------------------------------
# Per-connection session
# ---------------------------------------------------------------------------

class StreamSession:
    """
    State for one WebSocket connection.

    Inference runs in a thread pool so it never blocks the event loop.
    A flush fires every `send_interval` seconds, packaging whatever is
    currently in the buffer and sending it to the client.
    """

    def __init__(
        self,
        model:          MoveNetInference,
        exercise:       Optional[str] = None,
        send_interval:  float         = 2.0,
        window_seconds: float         = 2.0,
        sampled_images: int           = 4,
        jpeg_quality:   int           = 70,
        conf_threshold: float         = 0.3,
    ):
        self._model         = model
        self.exercise       = exercise
        self.send_interval  = send_interval
        self._buffer        = FrameBuffer(window_seconds, sampled_images,
                                          jpeg_quality, conf_threshold)
        self._batch_index   = 0
        self._last_flush    = 0.0

    def set_exercise(self, exercise: str) -> None:
        """Changing exercise clears the buffer so stale motion doesn't bleed in."""
        self.exercise = exercise
        self._buffer.clear()
        self._last_flush = 0.0

    def infer(self, jpeg_bytes: bytes) -> None:
        """Decode frame + run MoveNet. Called in thread pool — not async."""
        frame = _decode_jpeg(jpeg_bytes)
        if frame is None:
            return
        kp = self._model.predict(frame)
        self._buffer.add(frame, kp)

    def try_flush(self) -> Optional[ContextBatch]:
        """Return a batch if send_interval has elapsed, otherwise None."""
        now = time.monotonic()
        if now - self._last_flush < self.send_interval:
            return None
        self._last_flush = now
        self._batch_index += 1
        return self._buffer.build(self.exercise, self._batch_index, self._model)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_model: Optional[MoveNetInference] = None

def get_model() -> MoveNetInference:
    global _model
    if _model is None:
        _model = MoveNetInference.from_hub("lightning")
    return _model


app = FastAPI(title="MoveNet Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def load_model() -> None:
    get_model()
    print("MoveNet model loaded.")


@app.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint.

    Client → server messages:
      1. First message (text JSON) — session config:
             {"exercise": "squat", "send_interval": 2.0, "window_seconds": 2.0}
      2. Binary messages — JPEG frame bytes (every animation frame)
      3. Text messages at any time — update exercise:
             {"exercise": "deadlift"}

    Server → client messages:
      - ContextBatch JSON every send_interval seconds
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()

    # Parse config from first message — must be text JSON.
    try:
        first = await websocket.receive()
        config_raw = first.get("text") or (first.get("bytes") or b"{}").decode()
        config = json.loads(config_raw)
    except Exception as exc:
        print(f"Failed to parse config: {exc}")
        await websocket.close(code=1011)
        return

    session = StreamSession(
        model=get_model(),
        exercise=config.get("exercise"),
        send_interval=float(config.get("send_interval", 2.0)),
        window_seconds=float(config.get("window_seconds", 2.0)),
        sampled_images=int(config.get("sampled_images", 4)),
        jpeg_quality=int(config.get("jpeg_quality", 70)),
        conf_threshold=float(config.get("conf_threshold", 0.3)),
    )
    print(f"Session started — exercise={session.exercise}, "
          f"interval={session.send_interval}s, window={session.send_interval}s")

    try:
        while True:
            message = await websocket.receive()

            # Exercise change.
            if "text" in message:
                update = json.loads(message["text"])
                if "exercise" in update:
                    session.set_exercise(update["exercise"])
                    print(f"Exercise changed → {session.exercise}")
                continue

            jpeg_bytes = message.get("bytes")
            if not jpeg_bytes:
                continue

            # Run inference off the event loop.
            await loop.run_in_executor(_executor, session.infer, jpeg_bytes)

            # Flush the window to the client if it's time.
            batch = session.try_flush()
            if batch is not None:
                await websocket.send_text(batch.to_json())

    except WebSocketDisconnect:
        print(f"Client disconnected — exercise={session.exercise}")
    except Exception as exc:
        import traceback
        print(f"Session error: {exc}")
        traceback.print_exc()
        await websocket.close(code=1011)
