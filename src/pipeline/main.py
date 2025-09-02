import os
import numpy as np
import torch
import simpa as sp

from simpa import Tags
from numpy import random as rd
from msotrec.parametrization import ScannerIdEnum, FieldOfViewIdEnum

from volume_generation import create_simple_tissue, create_vessel_at_focus, create_3_vessels_around_focus, create_calf_tissue_gastrocnemius_transversal
from pipeline import perform_MB_reconstruction_in_SIMPA
from tools import set_logger

# FIXME temporary workaround for newest Intel architectures
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Turning off pip env verbosity
PIPENV_VERBOSITY=-1

def run_pipeline(
        random_seed: int = rd.randint(1, 1000),
        file_name_root: str = "test",
        volume_type: str = "gastrocnemius_transversal",
        spacing_mm: int | float = 0.1,
        volume_transducer_dim_mm: int | float = 75,
        volume_planar_dim_mm: int | float = 15,
        volume_height_mm: int | float = 30,
        wavelengths_nm: list = [700, 730, 760, 800, 850, 900],
        laser_energies_mj: list = [11.25, 12.54, 10.59, 11.04, 9.93, 9.75],
        acoustic_3d: bool = True,
        plot_convergence_MB: bool = False, 
        plot_l_curve_MB: bool = False,
        simulation_path: str = None
        ) -> None:

    """
    Runs a complete simulation pipeline aiming at obtaining Linear Unxmixing results from different imaging geometries.

    :param random_seed: seed used for reproducibility
    :param file_name_root: root of the storing file name
    :param volume_type: configuration chosen for the volume created
    :param spacing_mm: spacing between voxels (in mm). Can"t be higher than 0.22 in case of MB reconstruction. 
                        Usually picked in the interval [0.1, 0.2].
    :param volume_transducer_dim_mm: dimension (in mm) of the volume in the horizontal direction of the tranducer plane (x)
    :param volume_planar_dim_mm: dimension (in mm) of the volume in the direction out of the tranducer plane (y)
    :param volume_height_mm: dimension (in mm) of the volume in the vertical direction of the tranducer plane (z) 
                            without taking into account the probe part
    :param wavelengths_nm: list of wavelengths used for simulation
    :param laser_energies_mj: list of wavelength-dependant laser energies used in the optical forward simulation 
                            (calibration of system 2-21-03 Erlangen)
    :param acoustic_3d: if True, does the acoustic simulation in 3d, else, in 2D
    :param plot_convergence_MB: plotting MB reconstruction convergence plot if True
    :param plot_l_curve_MB: plotting MB reconstruction L-curve if True
    :param simulation_path: path to the simulation folder
    """

    ### SETTINGS ###

    # Definition of the logger to store the log
    volume_name = file_name_root + "_" + str(random_seed)
    output_log_name = "./data/logs/" + volume_name + ".log"
    logger = set_logger(output_log_name)

    # Definition of the recurrent target path
    target_path = simulation_path + "/" + volume_name + ".hdf5"

    # Initialize global settings and prepare for simulation pipeline
    general_settings = {
        Tags.RANDOM_SEED: random_seed,
        Tags.VOLUME_NAME: volume_name,
        Tags.SIMULATION_PATH: simulation_path,
        Tags.SIMPA_OUTPUT_FILE_PATH: target_path,
        Tags.SPACING_MM: spacing_mm,
        Tags.DIM_VOLUME_Z_MM: volume_height_mm,
        Tags.DIM_VOLUME_X_MM: volume_transducer_dim_mm,
        Tags.DIM_VOLUME_Y_MM: volume_planar_dim_mm,
        Tags.WAVELENGTHS: wavelengths_nm,
        Tags.GPU: True,
        Tags.US_GEL: False,
        Tags.IGNORE_QA_ASSERTIONS: False, # ignoring a negative pressure problem that doesn"t bother us here
        Tags.DO_FILE_COMPRESSION: True,
        Tags.DO_IPASC_EXPORT: False
    }
    settings = sp.Settings(general_settings)

    # Choice of the volume to reconstruct
    if volume_type == "simple_tissue":
        structures = create_simple_tissue()
    elif volume_type == "vessel_at_focus":
        structures = create_vessel_at_focus()
    elif volume_type == "3_vessels_around_focus":
        structures = create_3_vessels_around_focus()
    elif volume_type == "gastrocnemius_transversal":
        structures = create_calf_tissue_gastrocnemius_transversal()
    else:
        logger.error(f"Unknown configuration chosen for volume creation.")

    # Setting specific settings
    settings.set_volume_creation_settings({
        Tags.SIMULATE_DEFORMED_LAYERS: True,
        Tags.STRUCTURES: structures
    })
    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: os.getenv("MCX_BINARY_PATH"),
        Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: laser_energies_mj
    })
    settings.set_acoustic_settings({
        Tags.ACOUSTIC_SIMULATION_3D: acoustic_3d,
        Tags.ACOUSTIC_MODEL_BINARY_PATH: os.getenv("MATLAB_BINARY_PATH"),
        Tags.KWAVE_PROPERTY_ALPHA_POWER: 0.00,
        Tags.KWAVE_PROPERTY_SENSOR_RECORD: "p",
        Tags.KWAVE_PROPERTY_PMLInside: False,
        Tags.KWAVE_PROPERTY_PMLSize: [31, 32],
        Tags.KWAVE_PROPERTY_PMLAlpha: 1.5,
        Tags.KWAVE_PROPERTY_PlotPML: False,
        Tags.RECORDMOVIE: False,
        Tags.MOVIENAME: "visualization_log",
        Tags.ACOUSTIC_LOG_SCALE: True
    })
    settings["FieldOfViewCropping"] = {
        Tags.DATA_FIELD: [
            Tags.DATA_FIELD_ALPHA_COEFF,
            Tags.DATA_FIELD_DENSITY,
            Tags.DATA_FIELD_ANISOTROPY,
            Tags.DATA_FIELD_GRUNEISEN_PARAMETER,
            Tags.DATA_FIELD_ABSORPTION_PER_CM,
            Tags.DATA_FIELD_SCATTERING_PER_CM,
            Tags.DATA_FIELD_SEGMENTATION,
            Tags.DATA_FIELD_SPEED_OF_SOUND,
            Tags.DATA_FIELD_FLUENCE
            ]
    }
    settings["noise_initial_pressure"] = {
        Tags.NOISE_MEAN: 1,
        Tags.NOISE_STD: 0.01,
        Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
        Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
        Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
    }
    settings["noise_time_series"] = {
        Tags.NOISE_MEAN: 0,
        Tags.NOISE_STD: 0.01,
        Tags.NOISE_MODE: Tags.NOISE_MODE_ADDITIVE,
        Tags.DATA_FIELD: Tags.DATA_FIELD_TIME_SERIES_DATA,
        Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
    }

    # Definition and positioning of device
    device_position = np.array(
        [volume_transducer_dim_mm/2,
        volume_planar_dim_mm/2,
        0]
        )
    epsilon = 0
    field_of_view_extent = np.array(
        [-20.75 + epsilon, 
        20.75 + epsilon, 
        0,
        0,
        -20.75 + 9,
        20.75 + 9]
        ) # matching that of MB rec, the shift of 8 + 1 mm due to the placement of focus point + height of mediprene membrane
                                                # CAREFUL : to be modified again if addition of US gel !
                                                # Seems to be differently defined in 2D and 3D
    device = sp.MSOTAcuityEcho(device_position_mm=device_position, field_of_view_extent_mm=field_of_view_extent)
    detection_geometry = sp.CurvedArrayDetectionGeometry(
        pitch_mm=0.34,
        radius_mm=40,
        number_detector_elements=256,
        detector_element_width_mm=0.24,
        detector_element_length_mm=13,
        center_frequency_hz=3.96e6,
        bandwidth_percent=153, # modified compared to the default value of 55%
        sampling_frequency_mhz=40,
        angular_origin_offset=np.pi,
        device_position_mm=device_position,
        field_of_view_extent_mm=field_of_view_extent
        )
    device.set_detection_geometry(detection_geometry)
    device.update_settings_for_use_of_model_based_volume_creator(settings)


    ### SIMULATION ###
        
    # Run simulation pipeline for all wavelengths
    pipeline = [
        sp.ModelBasedAdapter(settings),
        sp.MCXAdapter(settings),
        sp.KWaveAdapter(settings),
        sp.FieldOfViewCropping(settings, settings_key=True)
    ]
    sp.simulate(pipeline, settings, device)


    ### MODEL-BASED RECONSTRUCTION ###

    # Checking that spacing lower than 0.22 (MB rec requirement)
    if spacing_mm > 0.22:
        raise ValueError(
            "Spatial spacing : {} used in SIMPA too high for MB reconstruction. Should be lower than 0.22."
            .format(spacing_mm))

    # Performing MB reconstruction
    perform_MB_reconstruction_in_SIMPA(
        target_path,
        volume_name=volume_name,
        settings=settings,
        field_of_view_id=FieldOfViewIdEnum.ITHERA_CLINICAL_416x416,
        scanner_id=ScannerIdEnum.ITHERA_CLINICAL_PTYPE218,
        wavelengths=wavelengths_nm,
        sampling_frequency_mhz=40,
        spacing_mm=spacing_mm,
        speed_of_sound_m_per_s=1540,
        laser_energies=torch.tensor(laser_energies_mj, dtype=torch.float64, device=torch.device("cuda:0")),
        dtype_simpa=np.float64, 
        dtype_mb=torch.float64,
        torch_device=torch.device("cuda:0"),
        logger=logger,
        perform_reconstruction=True, 
        plot_convergence=plot_convergence_MB, 
        plot_l_curve=plot_l_curve_MB
        )
    

    ### FOV CROPPING OF DATA FIELDS OF INTEREST ###

    # Redefinition of settings
    settings["FieldOfViewCropping"] = {
        Tags.DATA_FIELD: [
            Tags.DATA_FIELD_INITIAL_PRESSURE,
            Tags.DATA_FIELD_OXYGENATION,
            Tags.DATA_FIELD_BLOOD_VOLUME_FRACTION
            ]
    }

    field_of_view_extent = np.array(
        [-20.75 + epsilon, 
        20.75 + epsilon, 
        -1,
        1,
        -20.75 + 9,
        20.75 + 9]
        )
    
    device = sp.MSOTAcuityEcho(device_position_mm=device_position, field_of_view_extent_mm=field_of_view_extent)
    detection_geometry = sp.CurvedArrayDetectionGeometry(
        pitch_mm=0.34,
        radius_mm=40,
        number_detector_elements=256,
        detector_element_width_mm=0.24,
        detector_element_length_mm=13,
        center_frequency_hz=3.96e6,
        bandwidth_percent=153, # modified compared to the default value of 55%
        sampling_frequency_mhz=40,
        angular_origin_offset=np.pi,
        device_position_mm=device_position,
        field_of_view_extent_mm=field_of_view_extent
        )
    
    device.set_detection_geometry(detection_geometry)
    device.update_settings_for_use_of_model_based_volume_creator(settings)

    settings[Tags.CONTINUE_SIMULATION] = True

    # FOV cropping
    fov_cropping_data_of_interest = [sp.FieldOfViewCropping(settings, settings_key=True)]
    sp.simulate(fov_cropping_data_of_interest, settings, device)