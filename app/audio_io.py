from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Generator, Optional

import sounddevice as sd


@dataclass
class AudioStats:
    dropped_chunks: int = 0
    callback_errors: int = 0
    input_overflows: int = 0


class MicrophonePCM16Stream:
    """
    Stable microphone streamer for 16kHz/mono/PCM16 audio.
    Produces fixed-size byte chunks suitable for gRPC streaming APIs.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_ms: int = 50,
        queue_max_chunks: int = 100,
        device: Optional[int | str] = None,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("sample_rate must be 16000 for this streamer")
        if channels != 1:
            raise ValueError("channels must be 1 (mono)")
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be > 0")

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.frames_per_chunk = int(sample_rate * chunk_ms / 1000)
        self.bytes_per_chunk = self.frames_per_chunk * channels * 2  # int16 => 2 bytes
        self.device = device

        self._queue: Queue[Optional[bytes]] = Queue(maxsize=queue_max_chunks)
        self._stream: Optional[sd.RawInputStream] = None
        self._closed = True
        self.stats = AudioStats()

    def __enter__(self) -> MicrophonePCM16Stream:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        if self._stream is not None:
            return
        self._closed = False
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.frames_per_chunk,
            device=self.device,
            callback=self._callback,
            never_drop_input=False,
        )
        self._stream.start()

    def close(self) -> None:
        self._closed = True
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # Unblock any waiting consumer.
        try:
            self._queue.put_nowait(None)
        except Full:
            pass

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.stats.callback_errors += 1
            if getattr(status, "input_overflow", False):
                self.stats.input_overflows += 1

        # Copy out bytes to detach from PortAudio buffer lifetime.
        payload = bytes(indata)
        try:
            self._queue.put_nowait(payload)
        except Full:
            self.stats.dropped_chunks += 1
            # Drop oldest chunk first to keep latency bounded.
            try:
                _ = self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(payload)
            except Full:
                self.stats.dropped_chunks += 1

    def chunks(self, timeout_s: float = 1.0) -> Generator[bytes, None, None]:
        """
        Yield fixed-size PCM16 chunks.
        Stops when stream is closed.
        """
        carry = b""
        while True:
            if self._closed and self._queue.empty():
                return
            try:
                item = self._queue.get(timeout=timeout_s)
            except Empty:
                continue
            if item is None:
                return

            carry += item
            while len(carry) >= self.bytes_per_chunk:
                yield carry[: self.bytes_per_chunk]
                carry = carry[self.bytes_per_chunk :]


def microphone_chunks_16k(
    chunk_ms: int = 50,
    queue_max_chunks: int = 100,
    device: Optional[int | str] = None,
) -> Generator[bytes, None, None]:
    """
    Convenience generator for 16kHz mono PCM16 mic chunks.
    """
    with MicrophonePCM16Stream(
        sample_rate=16000,
        channels=1,
        chunk_ms=chunk_ms,
        queue_max_chunks=queue_max_chunks,
        device=device,
    ) as stream:
        yield from stream.chunks()
