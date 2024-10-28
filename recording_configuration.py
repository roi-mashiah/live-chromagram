import pyaudio
from dataclasses import dataclass


@dataclass
class RecordingConfiguration:
    sample_rate: int = 48000
    channels: int = 1
    bits_per_sample: int = 16
    sample_format: int = pyaudio.get_format_from_width(width=bits_per_sample // 8)  # width in bytes
    buffer_duration_sec: float = 2
    _buffer_size: int = int(buffer_duration_sec * sample_rate)
    numpy_dtype: str = f"int{bits_per_sample}"

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @buffer_size.getter
    def buffer_size(self) -> int:
        self._buffer_size = int(self.buffer_duration_sec * self.sample_rate)
        return self._buffer_size
