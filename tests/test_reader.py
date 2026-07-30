"""StreamReader tests."""

import threading
import time
import uuid

import numpy as np
import pytest

from arraystream.errors import Overrun
from arraystream.reader import StreamReader
from arraystream.writer import StreamWriter


def unique_stream_name(prefix: str = "reader") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def make_frames(n_frames: int, *, start: int = 0) -> np.ndarray:
    shape = (n_frames, 4, 8)
    frames = np.empty(shape, dtype=np.int32)
    for frame_index in range(n_frames):
        frames[frame_index].fill(start + frame_index)
    return frames


def test_read_batch_round_trip() -> None:
    name = unique_stream_name()
    writer = StreamWriter.create(
        name,
        frame_shape=(4, 8),
        dtype=np.int32,
        capacity_frames=8,
        max_batch_frames=1,
    )
    try:
        writer.write(make_frames(1, start=10))
        writer.write(make_frames(1, start=11))
        writer.write(make_frames(1, start=12))
        reader = StreamReader.attach(name, start="oldest")
        try:
            frames = reader.read(2, timeout=0)
            assert frames[0, 0, 0] == 10
            assert frames[1, 0, 0] == 11
        finally:
            reader.close()
    finally:
        writer.close()


def test_overrun_policies() -> None:
    name = unique_stream_name("overrun")
    writer = StreamWriter.create(
        name,
        frame_shape=(4, 8),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    error_reader = StreamReader.attach(name, start="oldest", on_overrun="error")
    oldest_reader = StreamReader.attach(name, start="oldest", on_overrun="oldest")
    latest_reader = StreamReader.attach(name, start="oldest", on_overrun="latest")
    try:
        for frame_index in range(6):
            writer.write(make_frames(1, start=frame_index))

        with pytest.raises(Overrun):
            error_reader.read(1, timeout=0)

        frame = oldest_reader.read(1, timeout=0)
        assert frame[0, 0, 0] == 2
        assert oldest_reader.dropped_frames > 0

        frame = latest_reader.read(1, timeout=0)
        assert frame[0, 0, 0] == 5
    finally:
        error_reader.close()
        oldest_reader.close()
        latest_reader.close()
        writer.close()


def test_read_view_rejects_invalid_batch_size() -> None:
    name = unique_stream_name("view")
    writer = StreamWriter.create(
        name,
        frame_shape=(2, 2),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    reader = StreamReader.attach(name, start="latest")
    try:
        writer.write(np.ones((1, 2, 2), dtype=np.int32))
        writer.write(np.full((1, 2, 2), 2, dtype=np.int32))
        with pytest.raises(ValueError, match="n_frames must be at least 1"):
            with reader.read_view(0, timeout=0):
                pass
    finally:
        reader.close()
        writer.close()


def test_invalid_constructor_args() -> None:
    name = unique_stream_name("args")
    writer = StreamWriter.create(
        name,
        frame_shape=(2, 2),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    segment = writer._segment
    with pytest.raises(ValueError, match="safety_frames"):
        StreamReader(segment, safety_frames=-1)
    with pytest.raises(ValueError, match="on_overrun"):
        StreamReader(segment, on_overrun="bad")  # type: ignore[arg-type]
    reader = StreamReader.attach(name, start="oldest")
    with pytest.raises(ValueError, match="position must be"):
        reader.seek("bad")  # type: ignore[arg-type]
    reader.close()
    writer.close()


def test_overrun_with_custom_message() -> None:
    error = Overrun(dropped_frames=2, oldest_valid_frame_index=5, message="boom")
    assert str(error) == "boom"


def test_timeout_when_frames_not_yet_available() -> None:
    name = unique_stream_name("timeout")
    writer = StreamWriter.create(
        name,
        frame_shape=(2, 2),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    reader = StreamReader.attach(name, start="oldest")
    try:
        with pytest.raises(TimeoutError, match="timed out"):
            reader.read(1, timeout=0)
        with pytest.raises(TimeoutError, match="blocking wait"):
            reader.read(1, timeout=None)
    finally:
        reader.close()
        writer.close()


def test_reader_close_is_idempotent() -> None:
    name = unique_stream_name("reader_close")
    writer = StreamWriter.create(
        name,
        frame_shape=(2, 2),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    reader = StreamReader.attach(name, start="oldest")
    try:
        assert reader.name == name
        writer.write(np.ones((1, 2, 2), dtype=np.int32))
        writer.write(np.full((1, 2, 2), 2, dtype=np.int32))
        assert reader.reader_frame_index == 0
    finally:
        reader.close()
        reader.close()
        writer.close()


def test_seek_and_metrics() -> None:
    name = unique_stream_name("metrics")
    writer = StreamWriter.create(
        name,
        frame_shape=(4, 8),
        dtype=np.int32,
        capacity_frames=8,
        max_batch_frames=1,
    )
    reader = StreamReader.attach(name, start="oldest")
    try:
        writer.write(make_frames(1, start=0))
        assert reader.available_frames == 0
        writer.write(make_frames(1, start=1))
        assert reader.available_frames >= 1
        assert reader.lag_frames >= 1
        writer.write(make_frames(1, start=2))
        reader.seek("latest")
        assert reader.reader_frame_index == writer.writer_frame_index - 1
    finally:
        reader.close()
        writer.close()


def test_context_manager_closes_reader() -> None:
    name = unique_stream_name("ctx")
    writer = StreamWriter.create(
        name,
        frame_shape=(2, 2),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    try:
        writer.write(np.ones((1, 2, 2), dtype=np.int32))
        writer.write(np.full((1, 2, 2), 2, dtype=np.int32))
        with StreamReader.attach(name, start="oldest") as reader:
            frames = reader.read(1, timeout=0)
            assert frames[0, 0, 0] == 1
    finally:
        writer.close()


def test_available_frames_uses_final_index_after_close() -> None:
    name = unique_stream_name("closed")
    writer = StreamWriter.create(
        name,
        frame_shape=(2, 2),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    reader = StreamReader.attach(name, start="oldest")
    try:
        writer.write(np.ones((1, 2, 2), dtype=np.int32))
        writer.write(np.full((1, 2, 2), 2, dtype=np.int32))
        writer.close()
        assert reader.available_frames == 2
    finally:
        reader.close()


def test_read_view_detects_in_flight_clobber() -> None:
    name = unique_stream_name("clobber")
    writer = StreamWriter.create(
        name,
        frame_shape=(4, 8),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    reader = StreamReader.attach(name, start="oldest", on_overrun="error")
    caught: list[BaseException] = []
    try:
        for frame_index in range(4):
            writer.write(make_frames(1, start=frame_index))
        reader.read(1, timeout=0)

        barrier = threading.Barrier(2)

        def slow_reader() -> None:
            try:
                with reader.read_view(1, timeout=0) as view:
                    barrier.wait(timeout=1)
                    time.sleep(0.05)
                    _ = view[0, 0, 0]
            except BaseException as exc:
                caught.append(exc)

        thread = threading.Thread(target=slow_reader)
        thread.start()
        barrier.wait(timeout=1)
        for frame_index in range(2, 8):
            writer.write(make_frames(1, start=frame_index))
        thread.join(timeout=2)
        assert caught and isinstance(caught[0], Overrun)
        assert reader.dropped_frames > 0
    finally:
        reader.close()
        writer.close()


def test_read_latest_returns_newest_frames() -> None:
    name = unique_stream_name("latest")
    writer = StreamWriter.create(
        name,
        frame_shape=(4, 8),
        dtype=np.int32,
        capacity_frames=8,
        max_batch_frames=1,
    )
    reader = StreamReader.attach(name, start="oldest")
    try:
        for frame_index in range(5):
            writer.write(make_frames(1, start=frame_index))
        frames = reader.read_latest(2, timeout=0)
        assert frames[0, 0, 0] == 2
        assert frames[1, 0, 0] == 3
    finally:
        reader.close()
        writer.close()


def test_read_latest_does_not_advance_reader_frame_index() -> None:
    name = unique_stream_name("latest_cursor")
    writer = StreamWriter.create(
        name,
        frame_shape=(4, 8),
        dtype=np.int32,
        capacity_frames=8,
        max_batch_frames=1,
    )
    reader = StreamReader.attach(name, start="oldest")
    try:
        for frame_index in range(5):
            writer.write(make_frames(1, start=frame_index))
        cursor_before = reader.reader_frame_index
        reader.read_latest(2, timeout=0)
        assert reader.reader_frame_index == cursor_before
        frames = reader.read(2, timeout=0)
        assert frames[0, 0, 0] == 0
        assert frames[1, 0, 0] == 1
    finally:
        reader.close()
        writer.close()


def test_read_latest_requires_enough_frames_in_stream() -> None:
    name = unique_stream_name("latest_wait")
    writer = StreamWriter.create(
        name,
        frame_shape=(2, 2),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    reader = StreamReader.attach(name, start="latest")
    try:
        with pytest.raises(TimeoutError, match="latest frame"):
            reader.read_latest(2, timeout=0)
        writer.write(np.ones((1, 2, 2), dtype=np.int32))
        with pytest.raises(TimeoutError, match="latest frame"):
            reader.read_latest(2, timeout=0)
    finally:
        reader.close()
        writer.close()


def test_read_latest_view_rejects_invalid_batch_size() -> None:
    name = unique_stream_name("latest_view")
    writer = StreamWriter.create(
        name,
        frame_shape=(2, 2),
        dtype=np.int32,
        capacity_frames=4,
        max_batch_frames=1,
    )
    reader = StreamReader.attach(name, start="latest")
    try:
        writer.write(np.ones((1, 2, 2), dtype=np.int32))
        with pytest.raises(ValueError, match="n_frames must be at least 1"):
            with reader.read_latest_view(0, timeout=0):
                pass
    finally:
        reader.close()
        writer.close()
