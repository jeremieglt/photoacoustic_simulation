import numpy as np
import math

def verify_image_shape(
        image : np.ndarray
    ) -> np.ndarray:

    """
    Verifies the shape of the images for generation of the selected data, in case some slices are kept around the central one.
    The images are therefore supposed to be 2D (only middle slice) or 3D (a few slices in the middle).

    :param image: image whose shape has to be verified

    :returns: image in the correct shape
    """

    # Number of dimensions of the image
    n_dim = len(image.shape)

    if n_dim == 3:
        image = image[:, math.ceil(image.shape[1] / 2), :]
    elif (n_dim < 2) or (n_dim > 3):
        raise ValueError("Uncorrect shape for the data field. Should be between 2 and 3, but here got {}.".format(n_dim))
    
    return image