import pyaudio
from dataclasses import dataclass


@dataclass
class RecordingConfiguration:
    sample_rate: int = 48000
    channels: int = 2
    bits_per_sample: int = 16
    sample_format: int = pyaudio.get_format_from_width(width=bits_per_sample // 8)  # width in bytes
    buffer_duration_sec: float = 2
    buffer_size: int = int(buffer_duration_sec * sample_rate)
    numpy_dtype: str = f"int{bits_per_sample}"
