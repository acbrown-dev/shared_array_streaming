"""Writer for a shared frame stream."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from arraystream.header import (
    FLAG_CLOSED,
    FLAG_WRITER_ALIVE,
    read_close_state,
    read_writer_frame_index,
    write_close_state,
    write_writer_frame_index,
)
from arraystream.segment import Segment

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike


def copy_frames_to_ring(
    ring: np.ndarray,
    *,
    capacity_frames: int,
    start_frame_index: int,
    frames: np.ndarray,
) -> None:
    n_frames = frames.shape[0]
    start_slot_index = start_frame_index % capacity_frames
    end_slot_index = start_slot_index + n_frames
    if end_slot_index <= capacity_frames:
        ring[start_slot_index:end_slot_index] = frames
        return

    first_chunk = capacity_frames - start_slot_index
    ring[start_slot_index:] = frames[:first_chunk]
    ring[: n_frames - first_chunk] = frames[first_chunk:]


def ring_batch_view(
    ring: np.ndarray,
    *,
    capacity_frames: int,
    start_frame_index: int,
    n_frames: int,
) -> np.ndarray:
    start_slot_index = start_frame_index % capacity_frames
    end_slot_index = start_slot_index + n_frames
    if end_slot_index <= capacity_frames:
        return ring[start_slot_index:end_slot_index]
    first_chunk = capacity_frames - start_slot_index
    return np.concatenate(
        (ring[start_slot_index:], ring[: n_frames - first_chunk]),
        axis=0,
    )


class StreamWriter:
    """Single-writer endpoint for a named frame stream."""

    def __init__(self, segment: Segment) -> None:
        self._segment = segment
        self._metadata = segment.metadata
        self._closed = False
        flags, _ = read_close_state(segment.buffer)
        write_close_state(
            segment.buffer,
            flags=flags | FLAG_WRITER_ALIVE,
            final_frame_index=0,
        )

    @classmethod
    def create(
        cls,
        name: str,
        *,
        frame_shape: tuple[int, ...],
        dtype: DTypeLike,
        capacity_frames: int,
        max_batch_frames: int = 1,
    ) -> StreamWriter:
        segment = Segment.create(
            name,
            frame_shape=frame_shape,
            dtype=dtype,
            capacity_frames=capacity_frames,
            max_batch_frames=max_batch_frames,
        )
        return cls(segment)

    @property
    def name(self) -> str:
        return self._segment.name

    @property
    def writer_frame_index(self) -> int:
        return read_writer_frame_index(self._segment.buffer)

    def write(self, frames: ArrayLike) -> None:
        frames_array = np.asarray(frames, dtype=self._metadata.dtype)
        if frames_array.ndim != len(self._metadata.frame_shape) + 1:
            raise ValueError(
                "frames must have shape (n_frames, *frame_shape); "
                f"expected {len(self._metadata.frame_shape) + 1} dimensions, "
                f"got {frames_array.ndim}"
            )
        if frames_array.shape[1:] != self._metadata.frame_shape:
            raise ValueError(
                f"frame shape mismatch: expected {self._metadata.frame_shape}, "
                f"got {frames_array.shape[1:]}"
            )

        n_frames = frames_array.shape[0]
        if n_frames < 1:
            raise ValueError("frames batch must contain at least one frame")
        if n_frames > self._metadata.max_batch_frames:
            raise ValueError(
                f"batch size {n_frames} exceeds max_batch_frames "
                f"{self._metadata.max_batch_frames}"
            )
        self._claim_frames(n_frames)
        start_frame_index = self.writer_frame_index - n_frames
        copy_frames_to_ring(
            self._segment.ring,
            capacity_frames=self._metadata.capacity_frames,
            start_frame_index=start_frame_index,
            frames=frames_array,
        )

    @contextmanager
    def reserve(self, n_frames: int) -> Iterator[np.ndarray]:
        if n_frames < 1:
            raise ValueError("n_frames must be at least 1")
        if n_frames > self._metadata.max_batch_frames:
            raise ValueError(
                f"n_frames {n_frames} exceeds max_batch_frames "
                f"{self._metadata.max_batch_frames}"
            )
        self._claim_frames(n_frames)
        start_frame_index = self.writer_frame_index - n_frames
        start_slot_index = start_frame_index % self._metadata.capacity_frames
        end_slot_index = start_slot_index + n_frames
        if end_slot_index <= self._metadata.capacity_frames:
            yield self._segment.ring[start_slot_index:end_slot_index]
            return

        temp = np.empty(
            (n_frames, *self._metadata.frame_shape), dtype=self._metadata.dtype
        )
        yield temp
        copy_frames_to_ring(
            self._segment.ring,
            capacity_frames=self._metadata.capacity_frames,
            start_frame_index=start_frame_index,
            frames=temp,
        )

    def _claim_frames(self, n_frames: int) -> None:
        if self._closed:
            raise RuntimeError("stream writer is closed")
        start_frame_index = self.writer_frame_index
        write_writer_frame_index(self._segment.buffer, start_frame_index + n_frames)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        final_frame_index = self.writer_frame_index
        write_close_state(
            self._segment.buffer,
            flags=FLAG_CLOSED,
            final_frame_index=final_frame_index,
        )
        del self._segment.ring
        self._segment.close()
        self._segment.unlink()

    def __enter__(self) -> StreamWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
