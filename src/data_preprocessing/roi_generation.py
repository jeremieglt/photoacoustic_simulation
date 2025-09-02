import os
import h5py
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from roi_selector import ROISelector

def generate_rois() -> None:

    """
    Allow to draw a ROI on each image of the selected dataset and store it in the corresponding file.
    """

    # Retrieving the paths for loading and storing the data
    env_path = Path("path_config.env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        selected_data_path = os.getenv("SELECTED_DATA_SAVE_DIRECTORY")
    else:
        raise FileNotFoundError(f"File {env_path} can't be found.")

    # Iterating the function to draw and store ROIs on all the source files
    for filename in os.listdir(selected_data_path):
        draw_and_store_roi(filename, selected_data_path)

def draw_and_store_roi(
        filename: str,
        path: str
    ) -> None:

    """
    Allows to draw a ROI on the fields from the selected file and store it there.
    A preview of the fields with a ROI drawn on them is also stored.

    :param filename: file name
    :param path: path where all the data are stored
    """

    # Copying the file from the source folder to the storing folder
    file = os.path.join(path, filename)

    # Doing the required operations on the newly created file
    with h5py.File(file, "r+") as f:

        # Loading the three fields
        p0 = f["initial pressure"][0, :, :] # only the first wavelength selected
        p0_rec = f["reconstruction"][0, :, :] # only the first wavelength selected
        so2 = f["oxygenation"][:]

        # Let the user draw a ROI on the initial pressure image
        selector = ROISelector(p0)
        mask = selector.get_mask()
        if mask is None:
            print(f"No ROI selected. Skipping file : {filename}")

        # Save ROI mask
        if "ROI" in f:
            del f["ROI"]
        f.create_dataset("ROI", data=mask.astype(np.uint8))

        # Create and save preview image with ROI highlighted
        prev_p0, prev_p0_rec, prev_so2 = create_roi_visualization(p0, mask), \
            create_roi_visualization(p0_rec, mask), create_roi_visualization(so2, mask)
        if "ROI preview" in f:
            del f["ROI preview"]
        prev = f.create_group("ROI preview")
        prev.create_dataset("initial pressure", data=prev_p0)
        prev.create_dataset("reconstruction", data=prev_p0_rec)
        prev.create_dataset("oxygenation", data=prev_so2)

        print(f"ROI and previews saved in {filename}")

def create_roi_visualization(
        image: np.array,
        mask: np.array, 
        fade_factor: float = 0
    ) -> None:

    """
    Creates a preview of what the ROI looks like on the image.

    :param image: image on which we draw the ROI
    :param mask: ROI mask
    :param fade_factor: 0 if we only want the ROI, non 0 only for visualization matters.
    """

    preview = image * mask + image * (1 - mask) * fade_factor

    return preview