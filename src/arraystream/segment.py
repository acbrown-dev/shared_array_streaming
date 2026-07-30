"""Shared memory segment lifecycle and frame ring view."""

from __future__ import annotations

import os
import sys
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

import numpy as np

from arraystream.header import (
    HEADER_NBYTES,
    HeaderMetadata,
    decode_header,
    encode_header,
    frame_nbytes_from_shape,
    segment_nbytes,
    slot_nbytes_from_frame,
    validate_create_params,
    validate_stream_name,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


def require_posix() -> None:
    if sys.platform == "win32":
        raise OSError("arraystream supports POSIX platforms only (Linux and macOS)")


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def _unregister_from_resource_tracker(shm: SharedMemory) -> None:
    resource_tracker.unregister(shm._name, "shared_memory")


def open_shared_memory(name: str, *, create: bool, size: int = 0) -> SharedMemory:
    require_posix()
    validate_stream_name(name)
    if create:
        return SharedMemory(name=name, create=True, size=size)
    if sys.version_info >= (3, 13):
        return SharedMemory(name=name, track=False)
    shm = SharedMemory(name=name)
    _unregister_from_resource_tracker(shm)
    return shm


def frame_ring_view(
    buffer: memoryview,
    *,
    capacity_frames: int,
    frame_shape: tuple[int, ...],
    dtype: np.dtype,
    slot_nbytes: int,
) -> np.ndarray:
    frame = np.empty(frame_shape, dtype=dtype)
    frame_strides = frame.strides
    return np.ndarray(
        shape=(capacity_frames, *frame_shape),
        dtype=dtype,
        buffer=buffer,
        offset=HEADER_NBYTES,
        strides=(slot_nbytes, *frame_strides),
    )


class Segment:
    """Owns a shared memory segment and its decoded header."""

    def __init__(
        self, shm: SharedMemory, metadata: HeaderMetadata, *, owns_segment: bool
    ) -> None:
        self._shm = shm
        self.metadata = metadata
        self._owns_segment = owns_segment
        self._buffer = shm.buf
        self.ring = frame_ring_view(
            self._buffer,
            capacity_frames=metadata.capacity_frames,
            frame_shape=metadata.frame_shape,
            dtype=metadata.dtype,
            slot_nbytes=metadata.slot_nbytes,
        )

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def buffer(self) -> memoryview:
        return self._buffer

    @classmethod
    def create(
        cls,
        name: str,
        *,
        frame_shape: tuple[int, ...],
        dtype: DTypeLike,
        capacity_frames: int,
        max_batch_frames: int,
    ) -> Segment:
        require_posix()
        validate_stream_name(name)
        dtype_obj = np.dtype(dtype)
        validate_create_params(
            frame_shape=frame_shape,
            dtype=dtype_obj,
            capacity_frames=capacity_frames,
            max_batch_frames=max_batch_frames,
        )

        frame_nbytes = frame_nbytes_from_shape(frame_shape, dtype_obj)
        slot_nbytes = slot_nbytes_from_frame(frame_nbytes)
        total_nbytes = segment_nbytes(capacity_frames, slot_nbytes)

        try:
            shm = open_shared_memory(name, create=True, size=total_nbytes)
        except OSError as exc:
            raise OSError(
                f"failed to create shared memory segment {name!r} "
                f"({total_nbytes} bytes)"
            ) from exc

        encode_header(
            shm.buf,
            capacity_frames=capacity_frames,
            slot_nbytes=slot_nbytes,
            frame_nbytes=frame_nbytes,
            frame_shape=frame_shape,
            dtype=dtype_obj,
            max_batch_frames=max_batch_frames,
            writer_pid=os.getpid(),
        )
        return cls(shm, decode_header(shm.buf), owns_segment=True)

    @classmethod
    def attach(cls, name: str) -> Segment:
        shm = open_shared_memory(name, create=False)
        metadata = decode_header(shm.buf)
        return cls(shm, metadata, owns_segment=False)

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        if self._owns_segment:
            self._shm.unlink()

    def writer_is_alive(self) -> bool:
        return process_is_alive(self.metadata.writer_pid)
