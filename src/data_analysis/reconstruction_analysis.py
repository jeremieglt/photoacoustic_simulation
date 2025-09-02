import os
import h5py
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
from simpa import Tags
from analysis_tools import normalize_image, compare_images, convert_rec_p0_to_array
from dotenv import load_dotenv
from pathlib import Path

def extract_pixel_wise_spectrum(
        data: np.ndarray, 
        position: tuple
    ) -> np.array:
    
    """
    Extracting the spectrum of a pixel at a certain position.
    
    :param data: reconstructed data, in the shape (num_wavelength, num_pixels_y, num_pixels_x)
    :param position: pixel-wise position of the pixel to analyse in the image in shape (x, z)
    
    :returns: spectrum of the pixel as the wanted position
    """

    # Creating the empty spectrum
    nb_wavelength = data.shape[0]

    # Extracting the spectrum
    spectrum = np.array([data[i, position[0], position[1]] for i in range(nb_wavelength)])

    return spectrum

def plot_reconstruction_spectrum(
        spectrum: np.array,
        position: tuple,
        wavelengths: list
    ) -> None:

    """
    Plotting the spectrum of a pixel.

    :param spectrum: spectrum of a pixel
    :param position: pixel-wise position of the pixel to analyse in the image in shape (x, z)
    :param wavelengths: list of wavleengthsused for unmixing
    """

    # Number of wavelengths used for unmixing
    nb_wl = len(wavelengths)

    # Plot the data
    plt.plot(spectrum, color='red')
    plt.title("Spectrum at position {} (pixels)".format(position))
    plt.xlabel("Wavelengths [nm]")
    plt.xticks(ticks=[i for i in range(nb_wl)], labels=[str(wl) for wl in wavelengths])
    plt.ylabel("Pressure amplitude [N/m²]")
    plt.show()

def normalize_compare_and_store_initial_and_rec_pressure(
        source_path: str,
        target_path: str
    ) -> None:

    """
    Normalizes, and then compares initial and reconstructed pressure images and finally stores them in a HDF5 file.

    :param source_path: path of the images to be loaded
    :param target_path: path where the relevant images for analysis are stored
    """

    p0 = sp.load_data_field(file_path=source_path, data_field=Tags.DATA_FIELD_INITIAL_PRESSURE, wavelength=800)
    p0 = p0[:, 10, :] # in the loaded file, multiple slices of p0 are kept around the central one. We only keep the middle slice here.
    p0_rec = sp.load_data_field(file_path=source_path, data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA, wavelength=800)

    p0_norm = normalize_image(p0)
    p0_rec_norm = normalize_image(p0_rec)

    mse_image, mae_image, mse, mae = compare_images(p0_norm, p0_rec_norm)

    # Saving the data fields
    with h5py.File(target_path, "w") as f:
        f.create_dataset("p0", data=p0, compression="gzip")
        f.create_dataset("p0 reconstructed", data=p0_rec, compression="gzip")
        f.create_dataset("p0 normalized", data=p0_norm, compression="gzip")
        f.create_dataset("p0 reconstructed normalized", data=p0_rec_norm, compression="gzip")
        f.create_dataset("MSE image", data=mse_image, compression="gzip")
        f.create_dataset("MAE image", data=mae_image, compression="gzip")
        f.create_dataset("MSE", data=mse)
        f.create_dataset("MAE", data=mae)

if __name__ == "__main__":

    # Loading environment path
    load_dotenv(Path("path_config.env"))
    raw_generated_data_path = os.getenv("RAW_DATA_SAVE_DIRECTORY")
    data_path = os.getenv("DATA_DIRECTORY")

    # Normalizing and comparing simulation data
    for i in range (1, 100):
        source_path = raw_generated_data_path + "/gt_" + str(i) + ".hdf5"
        target_path = data_path + "/image_range_analysis/range_gt_" + str(i) + ".hdf5"
        normalize_compare_and_store_initial_and_rec_pressure(source_path, target_path)

    # Analyzing pressure spectra
    simulated_data_path = raw_generated_data_path + "\gt_1.hdf5"
    rec = sp.load_data_field(simulated_data_path, data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA)
    rec_array = convert_rec_p0_to_array(rec)
    position = (117, 142)
    spectrum = extract_pixel_wise_spectrum(rec_array, position)
    plot_reconstruction_spectrum(spectrum, position, wavelengths=[700, 730, 760, 800, 850, 900])

    # FIXME creating functions to perform the two preceding workflows