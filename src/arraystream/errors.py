"""Stream errors."""


class StreamError(Exception):
    """Base error for arraystream."""


class StreamClosed(StreamError):
    """The writer has closed or crashed and no more frames are available."""


class Overrun(StreamError):
    """The reader fell behind and frames were overwritten."""

    def __init__(
        self,
        *,
        dropped_frames: int,
        oldest_valid_frame_index: int,
        message: str | None = None,
    ) -> None:
        self.dropped_frames = dropped_frames
        self.oldest_valid_frame_index = oldest_valid_frame_index
        if message is None:
            message = (
                f"reader overrun: {dropped_frames} frame(s) dropped; "
                f"oldest valid frame index is {oldest_valid_frame_index}"
            )
        super().__init__(message)
