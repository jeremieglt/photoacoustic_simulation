import numpy as np
import math

from scipy.stats import norm

def convert_rec_p0_to_array(
        data: dict
    ) -> np.ndarray:
    
    """
    Converting the reconstructed pressure dictionary (indexed on wavelengths) to a stacked array.

    :param data: data to unmix, dictionary of arrays in the shape (num_pixels_z, num_pixels_x) indexed on the wavelengths

    :returns: stacked array of the reconstructed arrays for each wavelength
    """

    # Defining the empty stack of sinograms
    nb_wls = len(data)
    first_wl = next(iter(data))
    first_array = data[first_wl]
    np_pxl_z, nb_pxl_x = first_array.shape
    stack = np.empty((nb_wls, np_pxl_z, nb_pxl_x))

    # Setting counter to 0
    i = 0

    # Adding elements to the empty array stack
    for array in data.values():
        stack[i, :, :] = array
        i = i + 1

    return stack

def normalize_image(
        image: np.ndarray
    ) -> np.ndarray:

    """
    Normalizing the image so that all the pixels fall in the intervall [0, 1]. 

    :param image: image to normalize

    :returns: normalized image
    """

    # Normalizing
    norm_image = (image - np.mean(image)) / (np.max(image) - np.min(image))

    return norm_image

def crop_and_pad_image_roi(
        image: np.ndarray,
        x_min: int,
        x_max: int,
        z_min: int,
        z_max: int
    ) -> np.ndarray:

    """
    Cropping an image to a region-of-interest (ROI) defined by its boundaries.
    The initial size of the image is kept for good comparison : only the ROI is kept non null.

    :param x_min: x lower boundary (in pixels)
    :param x_max: x higher boundary (in pixels)
    :param z_min: z lower boundary (in pixels)
    :param z_max: z higher boundary (in pixels)

    :returns: image of same size as the input, with a non null region (ROI) of size (x_max - x_min, z_max - z_min)
    """

    # Shape of the image
    image_shape = image.shape

    # Definition of a mask that will be filled with 1 in the ROI
    mask = np.zeros_like(image)

    # We consider that we only work with squared cropped images (so dim_x = dim_z)
    if len(image_shape) == 2:
        mask[x_min:x_max, z_min:z_max] = 1
    elif len(image_shape) == 3:
        if image_shape[1] == image_shape[2]: # case where we have multiple wavelengths : image_shape = (n_wavelengths, dim_x, dim_z)
            mask[:, x_min:x_max, z_min:z_max] = 1
        if image_shape[0] == image_shape[2]: # case where we have a non null y width : image_shape = (dim_x, dim_y, dim_z)
            mask[x_min:x_max, :, z_min:z_max] = 1
    elif len(image_shape) == 4: # case where we have multiple wavelengths and a non nully width : image_shape = (n_wavelengths, dim_x, dim_y, dim_z)
        mask[:, x_min:x_max, :, z_min:z_max] = 1
    else:
        raise ValueError("Uncorrect shape for the data field array. Should be between 2 and 4, but here got {}.".format(len(image_shape)))

    return image * mask

def crop_image_roi(
        image: np.ndarray,
        x_min: int,
        x_max: int,
        z_min: int,
        z_max: int
    ) -> np.ndarray:

    """
    Cropping an image to a region-of-interest (ROI) defined by its boundaries.

    :param x_min: x lower boundary (in pixels)
    :param x_max: x higher boundary (in pixels)
    :param z_min: z lower boundary (in pixels)
    :param z_max: z higher boundary (in pixels)

    :returns: an image of size (x_max - x_min, z_max - z_min)
    """

    # Shape of the image
    image_shape = image.shape

    # We consider that we only work with squared cropped images (so dim_x = dim_z)
    if len(image_shape) == 2:
        cropped_image = image[x_min:x_max, z_min:z_max]
    elif len(image_shape) == 3:
        if image_shape[1] == image_shape[2]: # case where we have multiple wavelengths : image_shape = (n_wavelengths, dim_x, dim_z)
            cropped_image = image[:, x_min:x_max, z_min:z_max]
        if image_shape[0] == image_shape[2]: # case where we have a non null y width : image_shape = (dim_x, dim_y, dim_z)
            cropped_image = image[x_min:x_max, :, z_min:z_max]
    elif len(image_shape) == 4: # case where we have multiple wavelengths and a non nully width : image_shape = (n_wavelengths, dim_x, dim_y, dim_z)
        cropped_image = image[:, x_min:x_max, :, z_min:z_max]
    else:
        raise ValueError("Uncorrect shape for the data field array. Should be between 2 and 4, but here got {}.".format(len(image_shape)))

    return cropped_image

def compare_images(
        image_1: np.ndarray,
        image_2: np.ndarray,
        roi: tuple = ()
    ) -> float:
    
    """
    Comparing two images using the Mean Squared Error (MSE).

    :param image_1: first image to compare
    :param image_2: second image to compare
    :param roi: tuple of coordinates of the region-of-interest (ROI), in shape (x_min, x_max, z_min, z_max).
                If nothing is provided, no cropping is done.

    :returns: Mean Squared Error (MSE) between the two images (inside ROI if specified)
    """

    # Crop images if ROI is defined
    if roi != ():
        x_min, x_max, y_min, y_max = roi
        image_1 = crop_image_roi(image_1, x_min, x_max, y_min, y_max)
        image_2 = crop_image_roi(image_2, x_min, x_max, y_min, y_max)

    # Computing errors
    mse_image = (image_1 - image_2) ** 2
    mae_image = np.abs(image_1 - image_2)
    mse = np.mean(mse_image)
    mae = np.mean(mae_image)

    return mse_image, mae_image, mse, mae

def average_image_y(
        image: np.ndarray,
        extent: int
    ) -> np.ndarray:

    """
    Computing the average of the frames of an image of size (dim_x, dim_y, dim_z) around its middle frame in the y direction.

    :param image: image to average in the y direction
    :param extent: number of frames that we average. Must be even (same number of frames on both sides of the central one).

    :returns: image of size (dim_x, dim_z) corresponding to the average of all the image frames in the y direction
    """

    # Verifying that the extent is even
    if extent % 2 != 0:
        raise ValueError("Chosen extent must be even.")

    # Determining the middle of the y direction
    image_shape = image.shape
    middle_y = math.ceil(image_shape[1] / 2)

    # Cropping the array to the chosen extent
    y_min, y_max = middle_y - int(extent / 2), middle_y + int(extent / 2)
    cropped_image = image[:, y_min:y_max, :]

    # Averaging
    avg_image = np.mean(cropped_image, axis=1)

    return avg_image

def gaussian_weighted_average_in_y(
        image: np.ndarray,
        extent: int,
        sigma: float
    ) -> np.ndarray:

    """
    Computing a Gaussian-weighted average of the frames of an image of size (dim_x, dim_y, dim_z) around its middle frame in the y direction.

    :param image: image to average in the y direction
    :param extent: number of frames that we average. Must be even (same number of frames on both sides of the central one).
    :param sigma: standard deviation of the Gaussian weights

    :returns: image of size (dim_x, dim_z) corresponding to the average of all the image frames in the y direction
    """

    # Verifying that the extent is even
    if extent % 2 != 0:
        raise ValueError("Chosen extent must be even.")
    
    # Determining the middle of the y direction
    image_shape = image.shape
    middle_y = math.ceil(image_shape[1] / 2)

    # Cropping the array to the chosen extent
    y_min, y_max = middle_y - int(extent / 2), middle_y + int(extent / 2)
    cropped_image = image[:, y_min:y_max+1, :]
    
    # Generating Gaussian weights centered around the middle slice
    y_indices = np.arange(y_max - y_min + 1)
    weights = norm.pdf(y_indices, loc=middle_y, scale=sigma)
    weights /= weights.sum() # normalizing the weights
    
    # Applying weighted averaging along the Y-axis
    weighted_avg_image = np.tensordot(cropped_image, weights, axes=(1, 0))
    
    return weighted_avg_image