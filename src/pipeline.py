import os
from dotenv import load_dotenv
from pathlib import Path

from pipeline import run_pipeline

if __name__ == "__main__":

    # Loading environment paths
    load_dotenv(Path("path_config.env"))

    # Running simulation pipeline
    run_pipeline(spacing_mm=0.2, acoustic_3d=False, simulation_path=os.getenv("DATA_DIRECTORY"))