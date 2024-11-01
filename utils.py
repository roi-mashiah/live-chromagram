import logging
import numpy as np
from numpy.fft import rfft
from colorama import Fore, Style
from colorama import init
from recording_configuration import RecordingConfiguration


def bytes_to_samples(raw_buffer, config: RecordingConfiguration):
    raw_samples = np.frombuffer(raw_buffer, dtype=config.numpy_dtype)
    reshaped_samples = raw_samples.reshape(-1, config.channels)
    normalized_samples = reshaped_samples / (2 ** (config.bits_per_sample - 1) - 1)
    normalized_samples = np.mean(normalized_samples, 1)
    return normalized_samples


class ColoredFormatter(logging.Formatter):
    # Map logging levels to colors
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA
    }

    def format(self, record):
        # Apply color based on the level
        color = self.COLORS.get(record.levelno, Fore.WHITE)
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)


def get_logger():
    init()
    logger = logging.getLogger("MyLogger")
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


class SpectrogramUtil:
    @staticmethod
    def pitch_to_frequency(pitch_number):
        """ Translate pitch to frequency where relative is A4 @ 440Hz"""
        return 440 * 0.5 ** ((69 - pitch_number) / 12)

    @staticmethod
    def get_frequency_mask(frequency_range, pitch):
        pitch_frequency = SpectrogramUtil.pitch_to_frequency(pitch)
        return (frequency_range <= pitch_frequency + 0.5) & (frequency_range >= pitch_frequency - 0.5)

    @staticmethod
    def calculate_log_frequency_spectrogram(power_spectrum, freq_axis):
        pitch_energies = np.zeros([1, 128])
        for pitch in range(128):
            mask = SpectrogramUtil.get_frequency_mask(freq_axis, pitch)
            pitch_energies[0, pitch] = np.sum(power_spectrum[mask])
        return pitch_energies

    @staticmethod
    def calculate_power_spectrum(x):
        n = 2 ** (len(x) - 1).bit_length()
        x_padded = np.concatenate([x, np.zeros(n - len(x))])
        win = np.hanning(n)
        dft = rfft(x_padded * win) / sum(win)
        power_spectrum = np.power(np.abs(dft), 2)
        return power_spectrum, n

    @staticmethod
    def calculate_pitch_energy_map(samples, sampling_rate):
        power_spectrum, n = SpectrogramUtil.calculate_power_spectrum(samples)
        f_res = sampling_rate / n
        freq_axis = np.arange(0, sampling_rate / 2 + f_res, f_res)
        log_freq_spectrum = SpectrogramUtil.calculate_log_frequency_spectrogram(power_spectrum, freq_axis)
        return log_freq_spectrum

    @staticmethod
    def calculate_chromogram(pitch_energy_map):
        notes = 12
        chromogram = np.zeros([1, notes])
        pitch_axis = np.arange(128)
        for n in range(notes):
            pitch_class_mask = (pitch_axis % notes) == n
            chromogram[0, n] = pitch_energy_map[0, pitch_class_mask].sum()
        return chromogram
