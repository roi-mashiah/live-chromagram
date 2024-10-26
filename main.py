from queue import Queue

import matplotlib.pyplot as plt
import numpy as np
import pyaudio

import utils
from input_device_manager import InputDeviceManager
from recording_configuration import RecordingConfiguration


def insert_buffer_to_queue(in_data, frame_count, time_info, status_flags):
    # put in_data to queue
    q.put(in_data)
    log.info(f"inserted to queue data of length {len(in_data)}")
    return None, pyaudio.paContinue


def plot_pitch_energy(pitch_energies):
    plt.clf()
    im = plt.imshow(np.log10(pitch_energies * 50 + 1), aspect='auto')
    plt.grid()
    plt.xlabel("Pitch")
    plt.pause(0.1)


def process_jobs_from_queue(config):
    m = 10
    im = np.zeros([m, 128])
    i = 0
    while True:
        raw_data = q.get()
        log.info("acquired data from queue, start processing...")
        samples = utils.bytes_to_samples(raw_data, config)
        current_pitch_energies = utils.SpectrogramUtil.process_time_samples(samples, config.sample_rate)
        im[i % m, :] = current_pitch_energies / np.max(current_pitch_energies)
        plot_pitch_energy(im)
        i += 1


if __name__ == '__main__':
    log = utils.get_logger()
    p = pyaudio.PyAudio()
    q = Queue()
    try:
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
        plt.show()
    except ValueError as device_error:
        log.error(f"choose different recording parameters: {device_error}")
    except Exception as ex:
        print(ex)
    finally:
        p.terminate()
        q.join()
