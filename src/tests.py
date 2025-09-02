import os
from dotenv import load_dotenv
from pathlib import Path
from numpy import random as rd

from tests import compare_msot_rec_and_kwave, visualise_device_simpa

if __name__ == "__main__":

    # Loading environment paths
    load_dotenv(Path("path_config.env"))
    simulation_path = os.getenv("DATA_DIRECTORY") + "msot_rec_test_data"

    # Seeding
    random_seed = rd.randint(1, 1000)

    compare_msot_rec_and_kwave(
        random_seed=random_seed,
        simulation_path=simulation_path, 
        target_path=simulation_path + "msot_rec_test_data/comparison_KW_MR_point_at_focus_" + str(random_seed) + ".png"
        )

    visualise_device_simpa(target_path=simulation_path + "/visualise_device_simpa.png")