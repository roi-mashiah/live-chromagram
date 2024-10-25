import logging
import numpy as np
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