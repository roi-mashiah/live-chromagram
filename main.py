from queue import Queue
from typing import Optional

import plotly.express as px
import numpy as np
import sounddevice
import pyaudio
import streamlit as st

import utils
from input_device_manager import InputDeviceManager
from recording_configuration import RecordingConfiguration

SAMPLING_RATES = [8000, 16000, 44100, 48000]
BITS_PER_SAMPLE = [8, 16, 24, 32]
CHROMA = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def insert_buffer_to_queue(in_data, frame_count, time_info, status_flags):
    q.put(in_data)
    log.info(f"inserted to queue data of length {len(in_data)}")
    log.debug(f"status flags - {status_flags}, time info - {time_info}")
    return (None, pyaudio.paContinue) if recording else (None, pyaudio.paAbort)


def process_jobs_from_queue(config):
    layout_dict = {"title": "Pitch Energies", "xaxis_title": "pitch", "yaxis_title": "time [sec]"}
    layout_dict_c = {"title": "Chromogram", "xaxis_title": "notes", "yaxis_title": "time [sec]"}
    m = 10
    pitch_energies = np.zeros([m, 128])
    chromagram = np.zeros([m, 12])
    i = 0

    fig = px.imshow(np.log10(pitch_energies * 50 + 1), aspect='auto')
    fig.update_layout(layout_dict)
    energies_ph.plotly_chart(fig, use_container_width=True, theme=None, key="pitch_energies")

    fig_c = px.imshow(np.log10(chromagram * 50 + 1), aspect='auto')
    fig_c.update_layout(layout_dict_c, xaxis={"tickmode": "array", "tickvals": list(range(12)), "ticktext": CHROMA})
    chromagram_ph.plotly_chart(fig_c, use_container_width=True, theme=None, key="chromogram")

    while True:
        if not recording:
            return
        raw_data = q.get()
        log.info("acquired data from queue, start processing...")
        samples = utils.bytes_to_samples(raw_data, config)
        current_pitch_energies = utils.SpectrogramUtil.calculate_pitch_energy_map(samples, config.sample_rate)
        current_chromagram = utils.SpectrogramUtil.calculate_chromogram(current_pitch_energies)

        chromagram[i % m, :] = current_chromagram / current_chromagram.max()
        pitch_energies[i % m, :] = current_pitch_energies / current_pitch_energies.max()

        fig = fig.update(data=[{"z": np.log10(pitch_energies * 50 + 1)}])
        fig_c = fig_c.update(data=[{"z": np.log10(chromagram * 50 + 1)}])
        energies_ph.plotly_chart(fig, use_container_width=True, theme=None)
        chromagram_ph.plotly_chart(fig_c, use_container_width=True, theme=None)
        i = i + 1 if i < m else 0


def init_sidebar_input_params():
    st.sidebar.subheader("Input Device")
    device = st.sidebar.selectbox("Devices", input_device_manager.input_devices)
    input_channel_options = set([device.max_input_channels for device in input_device_manager.input_devices])
    st.sidebar.subheader("Input Parameters")
    fs = st.sidebar.selectbox("Sampling Rate [Hz]", SAMPLING_RATES, index=0)
    frame_duration = st.sidebar.number_input("Frame Duration [sec]", min_value=0.1, max_value=4.0, value=2.0, step=0.1)
    bits_per_sample = st.sidebar.selectbox("Bits per sample", BITS_PER_SAMPLE, index=1)
    input_channels = st.sidebar.selectbox("Channels", input_channel_options, index=0)
    desired_config = RecordingConfiguration(sample_rate=fs,
                                            channels=input_channels,
                                            bits_per_sample=bits_per_sample,
                                            buffer_duration_sec=frame_duration)
    if p.is_format_supported(desired_config.sample_rate,
                             device.device_id,
                             desired_config.channels,
                             desired_config.sample_format):
        return desired_config
    else:
        st.sidebar.error("selected device doesn't support desired configurations")
        raise ValueError()


def start_recording():
    global recording, stream
    recording = True
    stream = p.open(
        rate=int(recording_parameters.sample_rate),
        channels=recording_parameters.channels,
        format=recording_parameters.sample_format,
        input=True,
        frames_per_buffer=recording_parameters.buffer_size,
        stream_callback=insert_buffer_to_queue
    )
    log.debug(recording_parameters)
    process_jobs_from_queue(recording_parameters)


def stop_recording():
    global recording, stream
    recording = False
    if stream:
        p.close(stream)
    return


if __name__ == '__main__':
    log = utils.get_logger()
    p = pyaudio.PyAudio()
    q = Queue()
    recording = False
    stream: Optional[pyaudio.Stream] = None
    st.set_page_config(layout="wide")
    try:
        input_device_manager = InputDeviceManager(p)
        recording_parameters = init_sidebar_input_params()
        log.debug(recording_parameters)
        input_device = input_device_manager.get_compatible_input_device(recording_parameters)
        st.title("Live Chroma features and Pitch energies")
        st.text(
            "This app demonstrates Chroma features and Pitch energy map in real time using the localhost's input audio devices.")
        st.divider()
        column_1, column_2 = st.columns(2)
        column_1.button("Start Recording", on_click=start_recording)
        column_2.button("Stop Recording", on_click=stop_recording)
        energies_ph = column_1.empty()
        chromagram_ph = column_2.empty()
    except ValueError as device_error:
        log.error(f"choose different recording parameters: {device_error}")
    except Exception as ex:
        log.error(ex)
    finally:
        p.terminate()
        q.join()
