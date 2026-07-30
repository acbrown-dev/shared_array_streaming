"""Reader for a shared frame stream."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import numpy as np

from arraystream.errors import Overrun
from arraystream.header import FLAG_CLOSED, read_close_state, read_writer_frame_index
from arraystream.segment import Segment
from arraystream.writer import ring_batch_view

StartPosition = Literal["latest", "oldest"]
OverrunPolicy = Literal["error", "oldest", "latest"]


class StreamReader:
    """Multi-reader endpoint for a named frame stream."""

    def __init__(
        self,
        segment: Segment,
        *,
        start: StartPosition = "latest",
        on_overrun: OverrunPolicy = "error",
        safety_frames: int = 0,
    ) -> None:
        if safety_frames < 0:
            raise ValueError("safety_frames must be non-negative")
        if on_overrun not in ("error", "oldest", "latest"):
            raise ValueError("on_overrun must be 'error', 'oldest', or 'latest'")

        self._segment = segment
        self._metadata = segment.metadata
        self._on_overrun = on_overrun
        self._safety_frames = safety_frames
        self._dropped_frames = 0
        self._reader_frame_index = 0
        self._closed = False

        if start == "latest":
            self.seek("latest")
        else:
            self.seek("oldest")

    @classmethod
    def attach(
        cls,
        name: str,
        *,
        start: StartPosition = "latest",
        on_overrun: OverrunPolicy = "error",
        safety_frames: int = 0,
    ) -> StreamReader:
        segment = Segment.attach(name)
        return cls(
            segment,
            start=start,
            on_overrun=on_overrun,
            safety_frames=safety_frames,
        )

    @property
    def name(self) -> str:
        return self._segment.name

    @property
    def reader_frame_index(self) -> int:
        return self._reader_frame_index

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def available_frames(self) -> int:
        readable_frame_index = self._readable_frame_index()
        return max(0, readable_frame_index - self._reader_frame_index)

    @property
    def lag_frames(self) -> int:
        writer_frame_index = self._writer_frame_index()
        return max(0, writer_frame_index - self._reader_frame_index)

    def seek(self, position: StartPosition) -> None:
        if position == "latest":
            self._reader_frame_index = self._readable_frame_index()
        elif position == "oldest":
            oldest_valid = self._oldest_valid_frame_index(self._writer_frame_index())
            self._reader_frame_index = max(0, oldest_valid)
        else:
            raise ValueError("position must be 'latest' or 'oldest'")

    def read(self, n_frames: int, *, timeout: float | None = None) -> np.ndarray:
        with self.read_view(n_frames, timeout=timeout) as view:
            return view.copy()

    def read_latest(self, n_frames: int, *, timeout: float | None = None) -> np.ndarray:
        with self.read_latest_view(n_frames, timeout=timeout) as view:
            return view.copy()

    @contextmanager
    def read_view(
        self, n_frames: int, *, timeout: float | None = None
    ) -> Iterator[np.ndarray]:
        if n_frames < 1:
            raise ValueError("n_frames must be at least 1")
        self._ensure_frames_available(n_frames, timeout=timeout)
        self._apply_overrun_policy_if_needed()
        start_frame_index = self._reader_frame_index
        view = ring_batch_view(
            self._segment.ring,
            capacity_frames=self._metadata.capacity_frames,
            start_frame_index=start_frame_index,
            n_frames=n_frames,
        )
        try:
            yield view
        finally:
            writer_frame_index = self._writer_frame_index()
            oldest_valid_frame_index = self._oldest_valid_frame_index(
                writer_frame_index
            )
            if start_frame_index < oldest_valid_frame_index:
                dropped = oldest_valid_frame_index - start_frame_index
                self._dropped_frames += dropped
                raise Overrun(
                    dropped_frames=dropped,
                    oldest_valid_frame_index=oldest_valid_frame_index,
                    message="read was clobbered while in progress",
                )
            self._reader_frame_index = start_frame_index + n_frames

    @contextmanager
    def read_latest_view(
        self, n_frames: int, *, timeout: float | None = None
    ) -> Iterator[np.ndarray]:
        if n_frames < 1:
            raise ValueError("n_frames must be at least 1")
        self._ensure_latest_frames_available(n_frames, timeout=timeout)
        start_frame_index = self._latest_start_frame_index(n_frames)
        view = ring_batch_view(
            self._segment.ring,
            capacity_frames=self._metadata.capacity_frames,
            start_frame_index=start_frame_index,
            n_frames=n_frames,
        )
        try:
            yield view
        finally:
            writer_frame_index = self._writer_frame_index()
            oldest_valid_frame_index = self._oldest_valid_frame_index(
                writer_frame_index
            )
            if start_frame_index < oldest_valid_frame_index:
                dropped = oldest_valid_frame_index - start_frame_index
                self._dropped_frames += dropped
                raise Overrun(
                    dropped_frames=dropped,
                    oldest_valid_frame_index=oldest_valid_frame_index,
                    message="read was clobbered while in progress",
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        del self._segment.ring
        self._segment.close()

    def __enter__(self) -> StreamReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _writer_frame_index(self) -> int:
        return read_writer_frame_index(self._segment.buffer)

    def _stream_is_closed(self) -> bool:
        flags, _ = read_close_state(self._segment.buffer)
        return bool(flags & FLAG_CLOSED)

    def _final_frame_index(self) -> int:
        _, final_frame_index = read_close_state(self._segment.buffer)
        return final_frame_index

    def _readable_frame_index(self) -> int:
        if self._stream_is_closed():
            return self._final_frame_index()
        return self._writer_frame_index() - self._metadata.max_batch_frames

    def _oldest_valid_frame_index(self, writer_frame_index: int) -> int:
        return writer_frame_index - self._metadata.capacity_frames + self._safety_frames

    def _apply_overrun_policy_if_needed(self) -> None:
        writer_frame_index = self._writer_frame_index()
        oldest_valid_frame_index = self._oldest_valid_frame_index(writer_frame_index)
        if self._reader_frame_index >= oldest_valid_frame_index:
            return

        dropped = oldest_valid_frame_index - self._reader_frame_index
        self._dropped_frames += dropped
        if self._on_overrun == "error":
            raise Overrun(
                dropped_frames=dropped,
                oldest_valid_frame_index=oldest_valid_frame_index,
            )
        if self._on_overrun == "oldest":
            self._reader_frame_index = oldest_valid_frame_index
            return
        self._reader_frame_index = self._readable_frame_index()

    def _latest_start_frame_index(self, n_frames: int) -> int:
        readable_frame_index = self._readable_frame_index()
        oldest_valid_frame_index = self._oldest_valid_frame_index(
            self._writer_frame_index()
        )
        return max(oldest_valid_frame_index, readable_frame_index - n_frames)

    def _available_latest_frames(self) -> int:
        readable_frame_index = self._readable_frame_index()
        oldest_valid_frame_index = self._oldest_valid_frame_index(
            self._writer_frame_index()
        )
        return max(0, readable_frame_index - max(0, oldest_valid_frame_index))

    def _ensure_frames_available(self, n_frames: int, *, timeout: float | None) -> None:
        if self.available_frames >= n_frames:
            return
        if timeout == 0:
            raise TimeoutError(
                f"timed out waiting for {n_frames} frame(s); "
                f"only {self.available_frames} available"
            )
        raise TimeoutError(
            f"only {self.available_frames} of {n_frames} frame(s) available; "
            "blocking wait is added in feat/waiting"
        )

    def _ensure_latest_frames_available(
        self, n_frames: int, *, timeout: float | None
    ) -> None:
        if self._available_latest_frames() >= n_frames:
            return
        if timeout == 0:
            raise TimeoutError(
                f"timed out waiting for {n_frames} latest frame(s); "
                f"only {self._available_latest_frames()} available"
            )
        raise TimeoutError(
            f"only {self._available_latest_frames()} of {n_frames} latest frame(s) "
            "available; blocking wait is added in feat/waiting"
        )
