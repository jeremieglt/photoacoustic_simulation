import logging
import sys
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

def set_logger(
        output_file_path: str = None
    ):

    """
    Sets a logger for the pipeline simulation. It will be both displayed in the terminal and saved in a log file.

    :param output_file_path: path to the log file that will store the logging info
    """

    # Configuring the logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # setting the minimum logging level to DEBUG

    # Creating a handler for writing logs to a file
    file_handler = logging.FileHandler(output_file_path)
    file_handler.setLevel(logging.DEBUG)  # Adjust as needed (DEBUG, INFO, etc.)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Creating a handler for outputting logs to the terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # Adjust as needed
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Adding handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Disabling logs from unwanted libraries (removing DEBUG and INFO level to keep only WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    return logger

def seed_everything(seed : int) -> None:

    """
    Seeds everything that is needed for reproducibility.

    :param seed: random seed

    :returns: None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multiple GPUs
    cudnn.deterministic = True
    cudnn.benchmark = False