import os
import numpy as np
import h5py
import simpa as sp

from simpa import Tags
from pathlib import Path
from dotenv import load_dotenv
from data_preprocessing import verify_image_shape

def generate_selected_data() -> None:

    """
    Generates the "selected" dataset by keeping only the necessary fields.
    """

    # Retrieving the paths for loading and storing the data
    env_path = Path("path_config.env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        raise FileNotFoundError(f"File {env_path} can't be found.")
    
    raw_data_path = os.getenv("SIMPA_SAVE_DIRECTORY")
    selected_data_path = os.getenv("SELECTED_DATA_SAVE_DIRECTORY")

    # Creating path to the storing folder
    selected_data_folder = Path(selected_data_path)
    selected_data_folder.mkdir(parents=True, exist_ok=True)

    # Iterating the function to draw and store ROIs on all the source files
    i = 0
    for filename in os.listdir(raw_data_path):
        i += 1
        extract_and_store_data(filename, raw_data_path, selected_data_path)
        if i == 2:
            break

def extract_and_store_data(
        filename : str,
        source_path : str,
        target_path : str
    ) -> None:

    """
    Extracts only the necesary data from the source file.

    :param filename: name of the file we want to extract from and store too (same name, usually "gt_n.hdf5")
    :param source_path: path to the source file containing unnecessary data
    :param target_path: path to the target file containing only the necessary data
    """

    # Copying the file from the source folder to the storing folder
    source_file = os.path.join(source_path, filename)
    target_file = os.path.join(target_path, filename)

    # If target file exists, delete it
    if os.path.exists(target_file):
        os.remove(target_file)

    # Extracting the so2 map and verifying its dimension
    so2 = sp.load_data_field(source_file, Tags.DATA_FIELD_OXYGENATION)
    so2 = verify_image_shape(so2)

    # Extracting the initial and reconstructed pressures
    settings = sp.load_data_field(source_file, Tags.SETTINGS)
    wavelengths = settings[Tags.WAVELENGTHS]
    n_wl = len(wavelengths)
    shape = (n_wl, so2.shape[0], so2.shape[1])
    p0 = np.empty(shape)
    p0_rec = np.empty(shape)

    for i in range(n_wl):

        wl = wavelengths[i]

        # Extracting the p0 map for the specific wavelength and verifying its dimension
        p0_wl = sp.load_data_field(source_file, Tags.DATA_FIELD_INITIAL_PRESSURE, str(wl))
        p0_wl = verify_image_shape(p0_wl)
        p0[i, :, :] = p0_wl

        # Extracting the p0_rec map for the specific wavelength and verifying its dimension
        p0_rec_wl = sp.load_data_field(source_file, Tags.DATA_FIELD_RECONSTRUCTED_DATA, str(wl))
        p0_rec_wl = verify_image_shape(p0_rec_wl)
        p0_rec[i, :, :] = p0_rec_wl

    # Storing the images in HDF5 format
    # Transposing for visualization matters
    with h5py.File(target_file, 'w') as f:
        f.create_dataset('initial pressure', data=p0)
        f.create_dataset('reconstruction', data=p0_rec)
        f.create_dataset('oxygenation', data=so2)