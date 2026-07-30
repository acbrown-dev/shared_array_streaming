"""StreamWriter tests."""

import uuid

import numpy as np
import pytest

from arraystream.header import FLAG_WRITER_ALIVE, read_close_state
from arraystream.segment import Segment
from arraystream.writer import StreamWriter, copy_frames_to_ring, ring_batch_view


def unique_stream_name(prefix: str = "writer") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def make_frames(n_frames: int, *, start: int = 0) -> np.ndarray:
    shape = (n_frames, 4, 8)
    frames = np.empty(shape, dtype=np.int32)
    for frame_index in range(n_frames):
        frames[frame_index].fill(start + frame_index)
    return frames


def test_write_and_claim_before_copy() -> None:
    name = unique_stream_name()
    with StreamWriter.create(
        name,
        frame_shape=(4, 8),
        dtype=np.int32,
        capacity_frames=8,
        max_batch_frames=2,
    ) as writer:
        assert writer.writer_frame_index == 0
        writer.write(make_frames(2, start=10))
        assert writer.writer_frame_index == 2

        segment = Segment.attach(name)
        try:
            assert segment.ring[0, 0, 0] == 10
            assert segment.ring[1, 0, 0] == 11
        finally:
            segment.close()


def test_reserve_zero_copy_and_wrap_fallback() -> None:
    name = unique_stream_name("reserve")
    writer = StreamWriter.create(
        name,
        frame_shape=(4, 8),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=2,
    )
    try:
        for batch_start in range(0, 6, 2):
            with writer.reserve(2) as slots:
                slots[:] = make_frames(2, start=batch_start)
        segment = Segment.attach(name)
        try:
            assert segment.ring[2, 0, 0] == 2
            assert segment.ring[0, 0, 0] == 4
        finally:
            segment.close()
    finally:
        writer.close()


def test_write_validates_batch_size() -> None:
    name = unique_stream_name("validate")
    with StreamWriter.create(
        name,
        frame_shape=(4, 8),
        dtype=np.int32,
        capacity_frames=8,
        max_batch_frames=1,
    ) as writer:
        with pytest.raises(ValueError, match="max_batch_frames"):
            writer.write(make_frames(2))


def test_copy_frames_to_ring_wraps() -> None:
    ring = np.zeros((4, 2, 2), dtype=np.int32)
    frames = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    copy_frames_to_ring(ring, capacity_frames=4, start_frame_index=3, frames=frames)
    assert ring[3, 0, 0] == 0
    assert ring[0, 0, 0] == 4


def test_ring_batch_view_wraps_to_copy() -> None:
    ring = np.arange(16, dtype=np.int32).reshape(4, 2, 2)
    view = ring_batch_view(ring, capacity_frames=4, start_frame_index=3, n_frames=2)
    assert view.shape == (2, 2, 2)
    assert view[0, 0, 0] == ring[3, 0, 0]


def test_writer_sets_alive_flag() -> None:
    name = unique_stream_name("alive")
    writer = StreamWriter.create(
        name,
        frame_shape=(2, 2),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    try:
        flags, _ = read_close_state(writer._segment.buffer)
        assert flags & FLAG_WRITER_ALIVE
        assert writer.name == name
    finally:
        writer.close()


def test_write_validates_shape_and_empty_batch() -> None:
    name = unique_stream_name("shape")
    writer = StreamWriter.create(
        name,
        frame_shape=(4, 8),
        dtype=np.int32,
        capacity_frames=8,
        max_batch_frames=2,
    )
    try:
        with pytest.raises(ValueError, match="dimensions"):
            writer.write(np.zeros((4, 8), dtype=np.int32))
        with pytest.raises(ValueError, match="frame shape mismatch"):
            writer.write(np.zeros((1, 2, 2), dtype=np.int32))
        with pytest.raises(ValueError, match="at least one frame"):
            writer.write(np.zeros((0, 4, 8), dtype=np.int32))
        with pytest.raises(ValueError, match="n_frames must be at least 1"):
            with writer.reserve(0):
                pass
    finally:
        writer.close()


def test_write_after_close_raises() -> None:
    name = unique_stream_name("closed")
    writer = StreamWriter.create(
        name,
        frame_shape=(2, 2),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.write(np.ones((1, 2, 2), dtype=np.int32))


def test_reserve_rejects_oversized_batch() -> None:
    name = unique_stream_name("reserve_max")
    writer = StreamWriter.create(
        name,
        frame_shape=(2, 2),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    try:
        with pytest.raises(ValueError, match="max_batch_frames"):
            with writer.reserve(2):
                pass
    finally:
        writer.close()


def test_ring_batch_view_contiguous_slice() -> None:
    ring = np.arange(16, dtype=np.int32).reshape(4, 2, 2)
    view = ring_batch_view(ring, capacity_frames=4, start_frame_index=1, n_frames=2)
    assert view.shape == (2, 2, 2)
    assert view[0, 0, 0] == ring[1, 0, 0]


def test_close_is_idempotent() -> None:
    name = unique_stream_name("idempotent")
    writer = StreamWriter.create(
        name,
        frame_shape=(2, 2),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    writer.close()
    writer.close()
