import numpy as np
import scipy
import simpa as sp
import h5py
import os
from enum import Enum, auto
from dotenv import load_dotenv
from pathlib import Path
from simpa import Tags
from analysis_tools import convert_rec_p0_to_array, compare_images, average_image_y, gaussian_weighted_average_in_y
from linear_unmixing_tools.retrieve_spectra_values import retrieve_absorption_values

class UnmixingId(Enum):
    ILABS = auto()
    LINEAR_NN_PRAHL_NO_900 = auto()
    LINEAR_NN_MOAVENI_NO_900 = auto()
    LINEAR_NN_PRAHL_NO_900_WITH_FC = auto()
    LINEAR_NN_MOAVENI_NO_900_WITH_FC = auto()

class ReferenceSpectraId(Enum):
    PRAHL = auto()
    MOAVENI = auto()
    TAKATANI = auto()
    OXY_AND_DEOXY_BLOOD = auto()
    HB_HBO2_H2O = auto()
    SIMPA = auto()

def load_spectra(
        normalize_spectra: bool, 
        source: ReferenceSpectraId
    ) -> tuple[np.ndarray, list[str]]:
    
    """
    Load the absorption spectra of deoxygenated and oxygenated haemoglobin for the wavelengths [700, 730, 760, 800, 850, 900].

    :param normalize_spectra: flag to normalize the spectra individually so that their L2 norms is 1
    :param source: identifier for reference spectra source
    
    :returns: absoption spectra, in the shape (2, 6); list of names, in the shape (2)
    """

    names = ["Hb", "HbO2"]

    if source == ReferenceSpectraId.PRAHL:
        # Source: https://omlc.org/spectra/hemoglobin/summary.html
        # The values are molar extinction coefficients in (cm^-1 / (moles/liter)).
        hbo2 = np.array([290.0, 390.0, 586.0, 816.0, 1058.0, 1198.0])
        hb = np.array([1794.28, 1102.2, 1548.52, 761.72, 691.32, 761.84])
        spectra = np.stack([hb, hbo2], axis=0)
        names = ["Hb", "HbO2"]

    elif source == ReferenceSpectraId.MOAVENI:
        # Source: https://omlc.org/spectra/hemoglobin/moaveni.html
        # The values are molar extinction coefficients in (cm^-1 / (moles/liter)).
        hbo2 = np.array([320.0, 400.0, 600.0, 800.0, 1060.0, 1200.0])
        hb = np.array([2160.0, 1500.0, 1720.0, 920.0, 800.0, 880.0])
        spectra = np.stack([hb, hbo2], axis=0)
        names = ["Hb", "HbO2"]

    elif source == ReferenceSpectraId.TAKATANI:
        # The values are molar extinction coefficients in (cm^-1 / (moles/liter)).
        # Source: https://omlc.org/spectra/hemoglobin/takatani.html
        hbo2 = np.array([290.0, 390.0, 586.0, 828.0, 1058.0, 1198.0])
        hb = np.array([2206.0, 1464.0, 1698.0, 930.0, 820.0, 890.0])
        spectra = np.stack([hb, hbo2], axis=0)
        names = ["Hb", "HbO2"]

    elif source == ReferenceSpectraId.OXY_AND_DEOXY_BLOOD:
        # Source: https://omlc.org/spectra/water/data/kou93b.txt
        h2o = np.array([0.006678, 0.019624, 0.028605, 0.022462, 0.041986, 0.064088])  # in cm^-1.
        hbo2 = np.array([290.0, 390.0, 586.0, 816.0, 1058.0, 1198.0])  # in (cm^-1 / (moles/liter)).
        hb = np.array([1794.28, 1102.2, 1548.52, 761.72, 691.32, 761.84])  # in (cm^-1 / (moles/liter)).
        hbo2_with_water = 2.303 * hbo2 * 150 / 64500 + h2o  # in cm^-1.
        hb_with_water = 2.303 * hb * 150 / 64500 + h2o  # in cm^-1.
        spectra = np.stack([hb_with_water, hbo2_with_water], axis=0)
        names = ["Hb+H2O", "HbO2+H2O"]

    elif source == ReferenceSpectraId.HB_HBO2_H2O:
        # Source: https://omlc.org/spectra/water/data/kou93b.txt
        h2o = np.array([0.006678, 0.019624, 0.028605, 0.022462, 0.041986, 0.064088])  # in cm^-1.
        hbo2 = np.array([290.0, 390.0, 586.0, 816.0, 1058.0, 1198.0])  # in (cm^-1 / (moles/liter)).
        hb = np.array([1794.28, 1102.2, 1548.52, 761.72, 691.32, 761.84])  # in (cm^-1 / (moles/liter)).
        hbo2 = 2.303 * hbo2 * 150 / 64500  # in cm^-1.
        hb = 2.303 * hb * 150 / 64500  # in cm^-1.
        spectra = np.stack([hb, hbo2, h2o], axis=0)
        names = ["Hb", "HbO2", "H2O"]

    elif source == ReferenceSpectraId.SIMPA:
        # Source: https://github.com/IMSY-DKFZ/simpa/tree/main/simpa/utils/libraries/absorption_spectra_data
        spectra_dict = retrieve_absorption_values(wavelengths=np.array([700, 730, 760, 800, 850, 900]))
        hbo2 = spectra_dict['Oxyhemoglobin'] # in cm^-1
        hb = spectra_dict['Deoxyhemoglobin'] # in cm^-1
        spectra = np.stack([hb, hbo2], axis=0)
        names = ["Hb", "HbO2"]

    else:
        raise ValueError(
            'Unknown reference spectra identifier `{}`. Valid identifiers are {}.'.format(
                source, [id for id in UnmixingId]))

    if normalize_spectra:
        spectra = spectra / np.linalg.norm(spectra, axis=1, keepdims=True)

    return spectra, names

def linear_unmixing(
        data: np.ndarray, 
        spectra: np.ndarray = None, 
        non_negative: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:

    """
    Performing Linear Unmixing on an image.
    
    :param data: data to unmix, in the shape (num_wavelength, num_pixels_z, num_pixels_x)
    :param spectra: spectra to unmix with, in the shape (num_chromophores, num_wavelengths)
    :param non_negative: only allow non-negative chromophore concentrations

    :returns:
        Chromophore concentrations, in the shape (num_chromophores, num_pixels_z, num_pixels_x)
        Residual norms per pixel ("|Ax-y|_2^2 / |y|_2^2"), in the shape (num_pixels_z, num_pixels_x)
    """

    data_orig_shape = data.shape
    if len(data_orig_shape) < 2 or len(data_orig_shape) > 3:
        raise ValueError("Invalid shape of data to unmix ({})".format(data.shape))

    data = np.reshape(data, (data_orig_shape[0], -1))
    n_wavelengths, n_pixels = data.shape

    if len(spectra.shape) != 2 or spectra.shape[1] != n_wavelengths:
        raise ValueError(
            "Invalid shape of spectra to unmix with ({}). Detected number of wavelength is {}."
            .format(spectra.shape, n_wavelengths))

    n_chromophores = spectra.shape[0]

    if non_negative:
        unmixed = np.empty((n_chromophores, n_pixels))
        residual_norms = np.empty(n_pixels)
        # "scipy.optimize.nnls" does not allow matrix-valued results, therefore we have to go pixel by pixel
        for i in range(n_pixels):
            unmixed[:, i], r = scipy.optimize.nnls(spectra.T, data[:, i], maxiter=1000000) # non-negative least-squares
            norm = np.linalg.norm(data[:, i], ord=2)
            residual_norms[i] = np.square(r / norm) if norm != 0 else 0
    else:
        unmixed, residuals, rank, s = np.linalg.lstsq(spectra.T, data, rcond=None)
        residual_norms = residuals / np.square(np.linalg.norm(data, axis=0))

    # Reshape output to the original shape (n_pixels) or (height, width)
    unmixed = np.reshape(unmixed, (n_chromophores, *data_orig_shape[1:]))
    residual_norms = np.reshape(residual_norms, (data_orig_shape[1:]))

    return unmixed, residual_norms

def compute_so2(
        unmixed: np.ndarray
        ):
    
    """
    Computes sO2 values from spectra of Hb and HbO2 retrieved after Linear Unmixing.

    :param unmixed: maps of Hb and HbO2 in this order, in shape (2, num_pixels_z, num_pixels_x)

    :returns: sO2 values, in shape (num_pixels) or (num_pixels_z, num_pixels_x)
    """

    # Calculate SO2 with handling division by zero or very small numbers
    denominator = unmixed[0, :, :] + unmixed[1, :, :]

    # Suppress the runtime warning
    np.seterr(divide='ignore', invalid='ignore')

    so2 = np.where(denominator != 0, unmixed[1, :, :] / denominator, 0)

    return so2

def unmix_and_store(
        source_path : str,
        target_path : str,
        roi : tuple = ()
    ) -> None:

    """
    Performs Linear Unmixing on one image and computes all the relevant metrics associated.

    :param source_path: path of the loaded data
    :param target_path: path where the data is stored
    :param roi: tuple of coordinates of the region-of-interest (ROI), in shape (x_min, x_max, z_min, z_max).
                If nothing is provided, no cropping is done.
    """

    # Loading segmentation
    seg = sp.load_data_field(source_path, data_field=Tags.DATA_FIELD_SEGMENTATION)

    # Loading p0 rec and convert it to a suitable shape
    p0_rec = sp.load_data_field(source_path, data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA)
    p0_rec_array = convert_rec_p0_to_array(p0_rec)

    # Loading ideal so2, and averaging it around 10 and 18 slices in the middle
    so2_GT = sp.load_data_field(source_path, data_field=Tags.DATA_FIELD_OXYGENATION)
    so2_GT_middle = so2_GT[:, 10, :] # selecting the middle slice
    so2_GT_avg_10 = average_image_y(so2_GT, extent=10) # averaging on 10 slices around the middle (11 slices)
    so2_GT_avg_18 = average_image_y(so2_GT, extent=18) # averaging on 18 slices around the middle (19 slices)

    # Gaussian averaging with std deviation of 3
    so2_GT_gauss_avg = gaussian_weighted_average_in_y(so2_GT, extent=10, sigma=3)

    # Loading spectra, linear unmixing and computing so2
    spectra, _ = load_spectra(normalize_spectra=True, source=ReferenceSpectraId.SIMPA) # Hb, HbO2, H2O
    unmixed, residuals = linear_unmixing(p0_rec_array, spectra, non_negative=True)
    so2_LU = compute_so2(unmixed)

    # Comparing the GT and LU so2
    so2_mse_image_middle, so2_mae_image_middle, so2_mse_middle, so2_mae_middle = compare_images(so2_LU, so2_GT_middle)
    so2_mse_image_avg_10, so2_mae_image_avg_10, so2_mse_avg_10, so2_mae_avg_10 = compare_images(so2_LU, so2_GT_avg_10)
    so2_mse_image_avg_18, so2_mae_image_avg_18, so2_mse_avg_18, so2_mae_avg_18 = compare_images(so2_LU, so2_GT_avg_18)
    so2_mse_image_gauss_avg, so2_mae_image_gauss_avg, so2_mse_gauss_avg, so2_mae_gauss_avg = compare_images(so2_LU, so2_GT_gauss_avg)

    # Computing the errors in the ROI
    if roi != ():
        _, _, so2_mse_middle_roi, so2_mae_middle_roi = compare_images(so2_LU, so2_GT_middle, roi)
        _, _, so2_mse_avg_10_roi, so2_mae_avg_10_roi = compare_images(so2_LU, so2_GT_avg_10, roi)
        _, _, so2_mse_avg_18_roi, so2_mae_avg_18_roi = compare_images(so2_LU, so2_GT_avg_18, roi)
        _, _, so2_mse_gauss_avg_roi, so2_mae_gauss_avg_roi = compare_images(so2_LU, so2_GT_gauss_avg, roi)

    # Save to HDF5 file
    with h5py.File(target_path, "w") as f:

        # Segmentation
        seg_group = f.create_group("seg")
        seg_group.create_dataset("seg", data=seg)

        # Reconstructed p0
        p0_group = f.create_group("p0")
        for wavelength, array in p0_rec.items():
            p0_group.create_dataset(wavelength, data=array)

        # Unmixing results and residuals
        LU_group = f.create_group("LU")
        LU_group.create_dataset("Hb", data=unmixed[0, :, :])
        LU_group.create_dataset("HbO2", data=unmixed[1, :, :])
        LU_group.create_dataset("LU residuals", data=residuals)
        LU_group.create_dataset("LU so2", data=so2_LU)

        # Ground truth so2s, error maps and global errors
        # Middle slice
        middle_group = f.create_group("middle slice")
        middle_group.create_dataset("GT so2", data=so2_GT_middle)
        middle_group.create_dataset("MSE image", data=so2_mse_image_middle)
        middle_group.create_dataset("MAE image", data=so2_mae_image_middle)
        middle_group.create_dataset("MSE", data=so2_mse_middle)
        middle_group.create_dataset("MAE", data=so2_mae_middle)
        if roi != ():
            middle_group.create_dataset("MSE ROI", data=so2_mse_middle_roi)
            middle_group.create_dataset("MAE ROI", data=so2_mae_middle_roi)
        # Average over 10 slices
        ten_slices_group = f.create_group("average 10 slices")
        ten_slices_group.create_dataset("GT so2", data=so2_GT_avg_10)
        ten_slices_group.create_dataset("MSE image", data=so2_mse_image_avg_10)
        ten_slices_group.create_dataset("MAE image", data=so2_mae_image_avg_10)
        ten_slices_group.create_dataset("MSE", data=so2_mse_avg_10)
        ten_slices_group.create_dataset("MAE", data=so2_mae_avg_10)
        if roi != ():
            ten_slices_group.create_dataset("MSE ROI", data=so2_mse_avg_10_roi)
            ten_slices_group.create_dataset("MAE ROI", data=so2_mae_avg_10_roi)
        # Average over 18 slices
        eighteen_slices_group = f.create_group("average 18 slices")
        eighteen_slices_group.create_dataset("GT so2", data=so2_GT_avg_18)
        eighteen_slices_group.create_dataset("MSE image", data=so2_mse_image_avg_18)
        eighteen_slices_group.create_dataset("MAE image", data=so2_mae_image_avg_18)
        eighteen_slices_group.create_dataset("MSE", data=so2_mse_avg_18)
        eighteen_slices_group.create_dataset("MAE", data=so2_mae_avg_18)
        if roi != ():
            eighteen_slices_group.create_dataset("MSE ROI", data=so2_mse_avg_18_roi)
            eighteen_slices_group.create_dataset("MAE ROI", data=so2_mae_avg_18_roi)
        # Gaussian average
        gaussian_group = f.create_group("gaussian average 10 slices")
        gaussian_group.create_dataset("GT so2", data=so2_GT_gauss_avg)
        gaussian_group.create_dataset("MSE image", data=so2_mse_image_gauss_avg)
        gaussian_group.create_dataset("MAE image", data=so2_mae_image_gauss_avg)
        gaussian_group.create_dataset("MSE", data=so2_mse_gauss_avg)
        gaussian_group.create_dataset("MAE", data=so2_mae_gauss_avg)
        if roi != ():
            gaussian_group.create_dataset("MSE ROI", data=so2_mse_gauss_avg_roi)
            gaussian_group.create_dataset("MAE ROI", data=so2_mae_gauss_avg_roi)

if __name__ == "__main__":

    # Loading environment path
    load_dotenv(Path("path_config.env"))
    raw_generated_data_path = os.getenv("RAW_DATA_SAVE_DIRECTORY")
    data_path = os.getenv("DATA_DIRECTORY")

    # Normalizing and comparing simulation data
    for i in range (1, 100):
        source_path = raw_generated_data_path + "/gt_" + str(i) + ".hdf5"
        target_path = data_path + "/linear_unmixing/LU_gt_" + str(i) + ".hdf5"
        unmix_and_store(source_path, target_path, roi=(107, 308, 250, 300))

    # FIXME Adapt code to selected data to analyze on ROI drawn by hand if needed.