import numpy as np
import matplotlib.pyplot as plt
import os
import simpa as sp
from dotenv import load_dotenv
from pathlib import Path

def compute_scattering_from_rayleigh_and_mie_theory(
        wavelengths: list = [],
        mus_at_500_nm: float = 1.0,
        fraction_rayleigh_scattering: float = 0.0,
        mie_power_law_coefficient: float = 0.0
    ) -> list:

    """
    Creates a scattering spectrum based on Rayleigh and Mie scattering theory.

    :param wavelengths: wavelengths used for simulation
    :param mus_at_500_nm: Scattering coefficient at 500 nm.
    :param fraction_rayleigh_scattering: Fraction of Rayleigh scattering.
    :param mie_power_law_coefficient: Power law coefficient for Mie scattering.

    :returns: scattering spectrum
    """
    
    scattering = (mus_at_500_nm * (fraction_rayleigh_scattering * (wavelengths / 500) ** 1e-4 +
                    (1 - fraction_rayleigh_scattering) * (wavelengths / 500) ** -mie_power_law_coefficient))
    
    return scattering

def plot_scattering(
        structure_names: list = ["background", "blood", "bone", "fat", "muscle", "epidermis", "dermis"], 
        colors: list = ["black", "red", "grey", "yellow", "orange", "brown", "pink"]
    ) -> None:
    
    """
    Plotting scattering spectra of molecules present in our tissues.

    :param structure_names: name of the structures we want to plot the spectrum from
    :param colors: colors of the respective graphs
    """

    # Number of spectra
    n_spectra = len(structure_names)

    # Empty list of spectra
    spectra = []

    # Loading environment path
    load_dotenv(Path("path_config.env"))
    source_path = os.getenv("DATA_DIRECTORY") + "literature_spectra/scattering_spectra_data/"
    target_path = os.getenv("DATA_DIRECTORY") + "literature_spectra/images/scattering_spectra"

    # Loading the spectra from the library
    for i in range(n_spectra - 2):
        spectrum = np.load(source_path + structure_names[i] + "_scattering.npz")
        spectra.append(spectrum)

    # Extract x (wavelengths) and y (scattering)
    wvl = spectra[0]["wavelengths"]
    values = []

    for spectrum in spectra:
        values.append(spectrum["values"])
    
    # Adding remaining values
    mus500_epi = sp.OpticalTissueProperties.MUS500_EPIDERMIS
    b_mie_epi = sp.OpticalTissueProperties.BMIE_EPIDERMIS
    f_ray_epi = sp.OpticalTissueProperties.FRAY_EPIDERMIS
    values_epi = compute_scattering_from_rayleigh_and_mie_theory(wvl, mus500_epi, f_ray_epi, b_mie_epi)
    values.append(values_epi)

    mus500_der = sp.OpticalTissueProperties.MUS500_DERMIS
    b_mie_der = sp.OpticalTissueProperties.BMIE_DERMIS
    f_ray_der = sp.OpticalTissueProperties.FRAY_DERMIS
    values_der = compute_scattering_from_rayleigh_and_mie_theory(wvl, mus500_der, f_ray_der, b_mie_der)
    values.append(values_der)

    # Plot the spectrum
    plt.figure(figsize=(8, 5))

    # Small precision
    labels = ["soft tissue scatterer", "blood scatterer (Hb and HbO$_{2}$)", "bone scatterer", "fat scatterer", 
              "muscle scatterer (with water)", "epidermal scatterer (with melanin)", "dermal scatterer"]

    for i in range(n_spectra):
        plt.plot(wvl, values[i], label=labels[i], color=colors[i])

    plt.xlabel("Wavelength $\lambda$ [nm]")
    plt.ylabel("Scattering coefficient $\mu_s$")
    plt.title("Scattering spectra")
    plt.legend()
    plt.grid()

    plt.savefig(target_path)

    plt.show()

def plot_absorption(
        structure_names: list = ["Water", "Oxyhemoglobin", "Deoxyhemoglobin", "Melanin", "Fat", "Skin_Baseline"], 
        colors: list = ["lightblue", "red", "blue", "brown", "yellow", "pink"]
    ) -> None:
    
    """
    Plotting absorption spectra of molecules present in our tissues.

    :param structure_names: name of the structures we want to plot the spectrum from
    :param colors: colors of the respective graphs
    """

    # Number of spectra
    n_spectra = len(structure_names)

    # Empty list of spectra
    spectra = []

    # Loading environment path
    load_dotenv(Path("path_config.env"))
    source_path = os.getenv("DATA_DIRECTORY") + "literature_spectra/absorption_spectra_data/"
    target_path = os.getenv("DATA_DIRECTORY") + "literature_spectra/images/absorption_spectra"

    # Loading the spectra from the library
    for i in range(n_spectra):
        spectrum = np.load(source_path + structure_names[i] + ".npz")
        spectra.append(spectrum)

    # Extract x (wavelengths) and y (scattering)
    wvl = spectra[0]["wavelengths"]
    values = []

    for spectrum in spectra:
        values.append(spectrum["values"])

    # Plot the spectrum
    plt.figure(figsize=(8, 5))

    # Little precision
    labels = ["water", "oxyhemoglobin", "deoxyhemoglobin", "melanin", "fat", "skin baseline"]

    for i in range(n_spectra):
        plt.plot(wvl, values[i], label=labels[i], color=colors[i])

    plt.xlabel("Wavelength $\lambda$ [nm]")
    plt.ylabel("Absorption coefficient $\mu_a$")
    plt.title("Absorption spectra")
    plt.legend()
    plt.grid()

    plt.savefig(target_path)

    plt.show()

def plot_anisotropy(
        structure_name: str = "Epidermis_Anisotropy", 
        color: str = "brown"
    ) -> None:

    """
    Plotting anisotropy spectrum of epidermis.

    :param structure_name: name of the structure we want to plot the spectrum from
    :param color: color of the respective graph
    """

    # Loading environment path
    load_dotenv(Path("path_config.env"))
    source_path = os.getenv("DATA_DIRECTORY") + "literature_spectra/anisotropy_spectra_data/"
    target_path = os.getenv("DATA_DIRECTORY") + "literature_spectra/images/anisotropy_spectra"

    # Loading the spectrum from the SIMPA library
    spectrum = np.load(source_path + structure_name + ".npz")
    
    # Extract x (wavelengths) and y (scattering)
    wvl = spectrum["wavelengths"]
    values = spectrum["values"]

    # Plot the spectrum
    plt.figure(figsize=(8, 5))

    # Little precision
    label = "epidermal anisotropy"

    plt.plot(wvl, values, label=label, color=color)
    plt.xlabel("Wavelength $\lambda$ [nm]")
    plt.ylabel("Anisotropy $g$")
    plt.title("Anisotropy spectrum")
    plt.legend()
    plt.grid()

    plt.savefig(target_path)

    plt.show()

if __name__ == "__main__":
    plot_scattering()
    plot_absorption()
    plot_anisotropy()