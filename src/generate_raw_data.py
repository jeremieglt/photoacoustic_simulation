import os
from dotenv import load_dotenv
from pathlib import Path
from pipeline import run_pipeline

def generate_gastrocnemius_transversal_dataset(
        n_iter : int = 100,
        last_id : int = 0
    ) -> None:

    """
    Script for generating a series of HDF5 files of optoacoustic simulation.
    The following scenario is chosen :
    - a scan of the gastrocnemius muscle in the transversal plane
    - 6 wavelengths between 700 and 900 nm are used for scanning
    - an isotropic resolution of 100 nm is used
    - the dimensions in (x, y, z) are (75, 15, 30) in mm
    - the acoustic simulation is made in 3D

    :param n_iter: number of generated files
    :param last_id: last id used for labeling the data
    """

    # Fixing the current id
    id = last_id

    # Running the pipeline for the number of times that was chosen
    while id < n_iter:

        id += 1

        file_name_root = "gt"
        wavelengths_nm = [700, 730, 760, 800, 850, 900]
        laser_energies_mj = [11.25, 12.54, 10.59, 11.04, 9.93, 9.75]

        # Loading environment paths
        load_dotenv(Path("path_config.env"))
        
        # Running simulation pipeline
        run_pipeline(
            random_seed=id,
            file_name_root=file_name_root,
            wavelengths_nm=wavelengths_nm,
            laser_energies_mj=laser_energies_mj,
            simulation_path=os.getenv("RAW_DATA_SAVE_DIRECTORY")
        )

    print(f"File gt_{id} generated with success !")

if __name__ == "__main__":
    generate_gastrocnemius_transversal_dataset(n_iter=500, last_id=222)