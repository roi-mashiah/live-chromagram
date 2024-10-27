import pyaudio
from typing import List
from dataclasses import dataclass
from recording_configuration import RecordingConfiguration


@dataclass
class InputDevice:
    device_name: str = ""
    device_id: int = -1
    max_input_channels: int = 0
    default_sample_rate: float = 8e3


class InputDeviceManager:
    def __init__(self, pa_object: pyaudio.PyAudio):
        self.pa_object = pa_object
        self.number_of_devices = self.pa_object.get_device_count()
        self.input_devices: List[InputDevice] = []
        self._get_input_devices()

    def _get_input_devices(self):
        for device_id in range(self.number_of_devices):
            device_info = self.pa_object.get_device_info_by_index(device_id)
            max_input_channels = device_info.get("maxInputChannels", 0)
            if max_input_channels > 0:
                self.input_devices.append(InputDevice(device_info["name"],
                                                      device_id,
                                                      max_input_channels,
                                                      device_info["defaultSampleRate"]))

    def get_compatible_input_device(self, recording_config: RecordingConfiguration):
        for candidate in self.input_devices:
            candidate_supports_parameters = self.pa_object.is_format_supported(rate=recording_config.sample_rate,
                                                                               input_device=candidate.device_id,
                                                                               input_channels=recording_config.channels,
                                                                               input_format=recording_config.sample_format)
            if candidate_supports_parameters:
                return candidate
        raise ValueError(f"couldn't find a compatible device for the desired parameters: {recording_config}")
