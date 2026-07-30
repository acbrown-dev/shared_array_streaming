"""Segment lifecycle tests."""

import uuid

import numpy as np
import pytest

from arraystream.header import HEADER_NBYTES, decode_header
from arraystream.segment import (
    Segment,
    frame_ring_view,
    open_shared_memory,
    process_is_alive,
    require_posix,
)


def unique_stream_name(prefix: str = "seg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def test_require_posix_rejects_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("arraystream.segment.sys.platform", "win32")
    with pytest.raises(OSError, match="POSIX"):
        require_posix()


def test_process_is_alive() -> None:
    assert process_is_alive(0) is False
    assert process_is_alive(__import__("os").getpid()) is True


def test_create_attach_and_ring_view() -> None:
    name = unique_stream_name()
    frame_shape = (4, 8)
    segment = Segment.create(
        name,
        frame_shape=frame_shape,
        dtype=np.int32,
        capacity_frames=8,
        max_batch_frames=2,
    )
    try:
        assert segment.ring.shape == (8, 4, 8)
        assert segment.ring.strides[0] >= 4 * 8 * 4

        attached = Segment.attach(name)
        try:
            metadata = decode_header(attached.buffer)
            assert metadata.frame_shape == frame_shape
            segment.ring[0].fill(7)
            assert attached.ring[0, 0, 0] == 7
        finally:
            attached.close()

        ring_only = frame_ring_view(
            segment.buffer,
            capacity_frames=8,
            frame_shape=frame_shape,
            dtype=np.dtype("int32"),
            slot_nbytes=segment.metadata.slot_nbytes,
        )
        assert ring_only.shape == (8, 4, 8)
        assert segment.name == name
        assert segment.writer_is_alive() is True
    finally:
        segment.close()
        segment.unlink()


def test_open_shared_memory_rejects_invalid_name() -> None:
    with pytest.raises(ValueError, match="empty"):
        open_shared_memory("", create=True, size=HEADER_NBYTES + 64)


def test_process_is_alive_dead_pid() -> None:
    assert process_is_alive(999_999_999) is False


def test_unlink_only_when_owner() -> None:
    name = unique_stream_name("unlink")
    owner = Segment.create(
        name,
        frame_shape=(2, 2),
        dtype=np.float32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    reader = Segment.attach(name)
    try:
        reader.unlink()
        owner.unlink()
    finally:
        owner.close()
        reader.close()
