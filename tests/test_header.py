"""Header layout tests."""

import numpy as np
import pytest

from arraystream.header import (
    FLAG_CLOSED,
    FLAG_WRITER_ALIVE,
    HEADER_NBYTES,
    decode_header,
    encode_header,
    frame_nbytes_from_shape,
    read_close_state,
    read_writer_frame_index,
    segment_nbytes,
    slot_nbytes_from_frame,
    validate_create_params,
    validate_stream_name,
    write_close_state,
    write_writer_frame_index,
    writer_frame_index_view,
)


def test_encode_decode_round_trip() -> None:
    buffer = bytearray(HEADER_NBYTES)
    frame_shape = (3, 5)
    dtype = np.dtype("float32")
    frame_nbytes = frame_nbytes_from_shape(frame_shape, dtype)
    slot_nbytes = slot_nbytes_from_frame(frame_nbytes)

    encode_header(
        buffer,
        capacity_frames=16,
        slot_nbytes=slot_nbytes,
        frame_nbytes=frame_nbytes,
        frame_shape=frame_shape,
        dtype=dtype,
        max_batch_frames=4,
        writer_pid=1234,
    )

    metadata = decode_header(buffer)
    assert metadata.capacity_frames == 16
    assert metadata.slot_nbytes == slot_nbytes
    assert metadata.frame_shape == frame_shape
    assert metadata.dtype == dtype
    assert metadata.max_batch_frames == 4
    assert metadata.writer_pid == 1234
    assert segment_nbytes(16, slot_nbytes) == HEADER_NBYTES + 16 * slot_nbytes


def test_slot_nbytes_aligns_to_64_bytes() -> None:
    frame_shape = (4, 8)
    dtype = np.dtype("int32")
    frame_nbytes = frame_nbytes_from_shape(frame_shape, dtype)
    slot_nbytes = slot_nbytes_from_frame(frame_nbytes)
    assert slot_nbytes % 64 == 0
    assert slot_nbytes >= frame_nbytes


def test_writer_frame_index_view() -> None:
    buffer = bytearray(HEADER_NBYTES)
    write_writer_frame_index(buffer, 42)
    assert read_writer_frame_index(buffer) == 42

    view = writer_frame_index_view(buffer)
    view[0] = np.uint64(99)
    assert read_writer_frame_index(buffer) == 99


def test_close_state_round_trip() -> None:
    buffer = bytearray(HEADER_NBYTES)
    flags = FLAG_WRITER_ALIVE | FLAG_CLOSED
    write_close_state(buffer, flags=flags, final_frame_index=99)
    assert read_close_state(buffer) == (flags, 99)


def test_validate_stream_name() -> None:
    validate_stream_name("camera0")
    with pytest.raises(ValueError, match="empty"):
        validate_stream_name("")
    with pytest.raises(ValueError, match="at most"):
        validate_stream_name("x" * 31)
    with pytest.raises(ValueError, match="start with"):
        validate_stream_name("/bad")
    with pytest.raises(ValueError, match="null"):
        validate_stream_name("bad\x00name")


def test_validate_create_params() -> None:
    dtype = np.dtype("float32")
    validate_create_params(
        frame_shape=(2, 2),
        dtype=dtype,
        capacity_frames=8,
        max_batch_frames=4,
    )
    with pytest.raises(ValueError, match="capacity_frames"):
        validate_create_params(
            frame_shape=(2, 2),
            dtype=dtype,
            capacity_frames=0,
            max_batch_frames=1,
        )
    with pytest.raises(ValueError, match="max_batch_frames must be at least"):
        validate_create_params(
            frame_shape=(2, 2),
            dtype=dtype,
            capacity_frames=4,
            max_batch_frames=0,
        )
    with pytest.raises(ValueError, match="max_batch_frames must not exceed"):
        validate_create_params(
            frame_shape=(2, 2),
            dtype=dtype,
            capacity_frames=4,
            max_batch_frames=8,
        )
    with pytest.raises(ValueError, match="frame_shape must not be empty"):
        validate_create_params(
            frame_shape=(),
            dtype=dtype,
            capacity_frames=4,
            max_batch_frames=1,
        )
    with pytest.raises(ValueError, match="at most"):
        validate_create_params(
            frame_shape=(1,) * 9,
            dtype=dtype,
            capacity_frames=4,
            max_batch_frames=1,
        )
    with pytest.raises(ValueError, match="positive"):
        validate_create_params(
            frame_shape=(0, 2),
            dtype=dtype,
            capacity_frames=4,
            max_batch_frames=1,
        )
    with pytest.raises(ValueError, match="object"):
        validate_create_params(
            frame_shape=(2,),
            dtype=np.dtype("O"),
            capacity_frames=4,
            max_batch_frames=1,
        )


def test_decode_rejects_bad_magic() -> None:
    buffer = bytearray(HEADER_NBYTES)
    with pytest.raises(ValueError, match="not an arraystream"):
        decode_header(buffer)


def test_decode_rejects_unsupported_version() -> None:
    buffer = bytearray(HEADER_NBYTES)
    encode_header(
        buffer,
        capacity_frames=4,
        slot_nbytes=64,
        frame_nbytes=64,
        frame_shape=(8,),
        dtype=np.dtype("float32"),
        max_batch_frames=1,
        writer_pid=1,
    )
    buffer[9] = 99
    with pytest.raises(ValueError, match="unsupported arraystream format version"):
        decode_header(buffer)


def test_decode_rejects_header_size_mismatch() -> None:
    buffer = bytearray(HEADER_NBYTES)
    encode_header(
        buffer,
        capacity_frames=4,
        slot_nbytes=64,
        frame_nbytes=64,
        frame_shape=(8,),
        dtype=np.dtype("float32"),
        max_batch_frames=1,
        writer_pid=1,
    )
    buffer[12:16] = (2048).to_bytes(4, "little")
    with pytest.raises(ValueError, match="header size mismatch"):
        decode_header(buffer)
