import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path

def retrieve_absorption_values(
        chromophores: str = ["Water", "Oxyhemoglobin", "Deoxyhemoglobin"],
        wavelengths: str = [700, 730, 760, 800, 850, 900]
    ) -> dict:
    
    """
    Retrieving values of absorption for the wavelengths chosen for unmixing.

    :param chromophores: name of the chromophores we want to plot the spectrum from
    :param wavelengths: wavelengths chosen for unmixing

    :returns: dict of values for absorption at the wavelengths chosen for unmixing for every chromophore
    """

    # Loading environment path
    load_dotenv(Path("path_config.env"))
    source_path = os.getenv("DATA_DIRECTORY") + "literature_spectra/absorption_spectra_data/"
    
    # Dictionary definition
    spectra = {
        chromophore: [] for chromophore in chromophores
    }

    # Loading the spectra from the SIMPA library
    for chromophore in chromophores:
        spectrum_data = np.load(source_path + chromophore + ".npz")
        wavelengths_tot = spectrum_data["wavelengths"]
        relevant_indices = np.where(np.isin(wavelengths_tot, wavelengths))[0]
        values = spectrum_data["values"]
        relevant_values = values[relevant_indices] * 64500 / (150 * 2.303)
        spectra[chromophore] = list(relevant_values)

    return spectra