from queue import Queue

import pyaudio
import numpy as np

import utils
from input_device_manager import InputDeviceManager
from recording_configuration import RecordingConfiguration


def insert_buffer_to_queue(in_data, frame_count, time_info, status_flags):
    # put in_data to queue
    q.put(in_data)
    log.info(f"inserted to queue data of length {len(in_data)}")
    return None, pyaudio.paContinue


def process_jobs_from_queue(config):
    while True:
        raw_data = q.get()
        log.info("acquired data from queue, start processing...")
        samples = utils.bytes_to_samples(raw_data, config)


if __name__ == '__main__':
    p = pyaudio.PyAudio()
    q = Queue()
    log = utils.get_logger()

    recording_parameters = RecordingConfiguration()
    input_device = InputDeviceManager(p).get_compatible_input_device(recording_parameters)

    input_stream = p.open(
        rate=int(recording_parameters.sample_rate),
        channels=recording_parameters.channels,
        format=recording_parameters.sample_format,
        input=True,
        frames_per_buffer=recording_parameters.buffer_size,
        stream_callback=insert_buffer_to_queue
    )
    process_jobs_from_queue(recording_parameters)

    p.terminate()
