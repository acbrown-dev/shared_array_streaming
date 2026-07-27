"""Self-describing segment header layout and accessors."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"SASTREAM"
FORMAT_VERSION = 1
HEADER_NBYTES = 4096
MAX_FRAME_NDIM = 8
DTYPE_FIELD_NBYTES = 64
MAX_STREAM_NAME_LEN = 30
SLOT_ALIGNMENT = 64

FLAG_WRITER_ALIVE = 1 << 0
FLAG_CLOSED = 1 << 1

_OFFSET_MAGIC = 0
_OFFSET_FRAME_META = 40
_OFFSET_FRAME_SHAPE = 56
_OFFSET_DTYPE = 120
_OFFSET_WRITER_FRAME_INDEX = 256
_OFFSET_CLOSE_STATE = 320

_STRUCT_PREFIX = struct.Struct("<8s2I3I")
_STRUCT_FRAME_META = struct.Struct("<4I")
_STRUCT_CLOSE_STATE = struct.Struct("<IQ")


@dataclass(frozen=True, slots=True)
class HeaderMetadata:
    capacity_frames: int
    slot_nbytes: int
    frame_nbytes: int
    frame_ndim: int
    max_batch_frames: int
    writer_pid: int
    frame_shape: tuple[int, ...]
    dtype: np.dtype


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def frame_nbytes_from_shape(frame_shape: tuple[int, ...], dtype: np.dtype) -> int:
    item_count = 1
    for dim in frame_shape:
        item_count *= dim
    return int(item_count * dtype.itemsize)


def slot_nbytes_from_frame(frame_nbytes: int) -> int:
    return align_up(frame_nbytes, SLOT_ALIGNMENT)


def validate_stream_name(name: str) -> None:
    if not name:
        raise ValueError("stream name must not be empty")
    if len(name) > MAX_STREAM_NAME_LEN:
        raise ValueError(
            f"stream name must be at most {MAX_STREAM_NAME_LEN} characters, "
            f"got {len(name)!r}"
        )
    if name.startswith("/"):
        raise ValueError("stream name must not start with '/'")
    if "\x00" in name:
        raise ValueError("stream name must not contain null bytes")


def validate_create_params(
    *,
    frame_shape: tuple[int, ...],
    dtype: np.dtype,
    capacity_frames: int,
    max_batch_frames: int,
) -> None:
    if capacity_frames < 1:
        raise ValueError("capacity_frames must be at least 1")
    if max_batch_frames < 1:
        raise ValueError("max_batch_frames must be at least 1")
    if max_batch_frames > capacity_frames:
        raise ValueError("max_batch_frames must not exceed capacity_frames")
    if not frame_shape:
        raise ValueError("frame_shape must not be empty")
    if len(frame_shape) > MAX_FRAME_NDIM:
        raise ValueError(f"frame_shape may have at most {MAX_FRAME_NDIM} dimensions")
    for dim in frame_shape:
        if dim < 1:
            raise ValueError("frame_shape dimensions must be positive")
    if dtype.hasobject:
        raise ValueError("dtype must not contain Python objects")


def encode_header(
    buffer: memoryview,
    *,
    capacity_frames: int,
    slot_nbytes: int,
    frame_nbytes: int,
    frame_shape: tuple[int, ...],
    dtype: np.dtype,
    max_batch_frames: int,
    writer_pid: int,
) -> None:
    dtype_str = dtype.str.encode("ascii")
    if len(dtype_str) >= DTYPE_FIELD_NBYTES:
        raise ValueError("dtype string is too long for the header")

    _STRUCT_PREFIX.pack_into(
        buffer,
        _OFFSET_MAGIC,
        MAGIC,
        FORMAT_VERSION,
        HEADER_NBYTES,
        capacity_frames,
        slot_nbytes,
        frame_nbytes,
    )
    _STRUCT_FRAME_META.pack_into(
        buffer,
        _OFFSET_FRAME_META,
        len(frame_shape),
        len(dtype_str),
        max_batch_frames,
        writer_pid,
    )

    shape_field = np.zeros(MAX_FRAME_NDIM, dtype=np.uint64)
    shape_field[: len(frame_shape)] = frame_shape
    buffer[_OFFSET_FRAME_SHAPE : _OFFSET_FRAME_SHAPE + shape_field.nbytes] = (
        shape_field.tobytes()
    )

    dtype_field = bytearray(DTYPE_FIELD_NBYTES)
    dtype_field[: len(dtype_str)] = dtype_str
    buffer[_OFFSET_DTYPE : _OFFSET_DTYPE + DTYPE_FIELD_NBYTES] = dtype_field

    writer_frame_index_view(buffer)[0] = np.uint64(0)
    write_close_state(buffer, flags=0, final_frame_index=0)


def decode_header(buffer: memoryview) -> HeaderMetadata:
    magic, format_version, header_nbytes = _STRUCT_PREFIX.unpack_from(
        buffer, _OFFSET_MAGIC
    )[:3]
    if magic != MAGIC:
        raise ValueError("not an arraystream segment")
    if format_version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported arraystream format version {format_version}, "
            f"expected {FORMAT_VERSION}"
        )
    if header_nbytes != HEADER_NBYTES:
        raise ValueError("header size mismatch")

    capacity_frames, slot_nbytes, frame_nbytes = _STRUCT_PREFIX.unpack_from(
        buffer, _OFFSET_MAGIC
    )[3:]
    frame_ndim, dtype_len, max_batch_frames, writer_pid = (
        _STRUCT_FRAME_META.unpack_from(buffer, _OFFSET_FRAME_META)
    )

    shape_field = np.frombuffer(
        buffer, dtype=np.uint64, count=MAX_FRAME_NDIM, offset=_OFFSET_FRAME_SHAPE
    )
    frame_shape = tuple(int(dim) for dim in shape_field[:frame_ndim])

    dtype_bytes = bytes(buffer[_OFFSET_DTYPE : _OFFSET_DTYPE + dtype_len])
    dtype = np.dtype(dtype_bytes.decode("ascii"))

    return HeaderMetadata(
        capacity_frames=capacity_frames,
        slot_nbytes=slot_nbytes,
        frame_nbytes=frame_nbytes,
        frame_ndim=frame_ndim,
        max_batch_frames=max_batch_frames,
        writer_pid=writer_pid,
        frame_shape=frame_shape,
        dtype=dtype,
    )


def writer_frame_index_view(buffer: memoryview) -> np.ndarray:
    return np.frombuffer(
        buffer, dtype=np.uint64, count=1, offset=_OFFSET_WRITER_FRAME_INDEX
    )


def read_writer_frame_index(buffer: memoryview) -> int:
    return int(writer_frame_index_view(buffer)[0])


def write_writer_frame_index(buffer: memoryview, writer_frame_index: int) -> None:
    writer_frame_index_view(buffer)[0] = np.uint64(writer_frame_index)


def read_close_state(buffer: memoryview) -> tuple[int, int]:
    flags, final_frame_index = _STRUCT_CLOSE_STATE.unpack_from(
        buffer, _OFFSET_CLOSE_STATE
    )
    return flags, int(final_frame_index)


def write_close_state(
    buffer: memoryview, *, flags: int, final_frame_index: int
) -> None:
    _STRUCT_CLOSE_STATE.pack_into(buffer, _OFFSET_CLOSE_STATE, flags, final_frame_index)


def segment_nbytes(capacity_frames: int, slot_nbytes: int) -> int:
    return HEADER_NBYTES + capacity_frames * slot_nbytes
