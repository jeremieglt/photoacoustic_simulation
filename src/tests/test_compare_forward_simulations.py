import torch
import os
import numpy as np
import simpa as sp
import logging
import matplotlib.pyplot as plt

from simpa import Tags
from simpa.io_handling import save_data_field
from msotrec.parametrization import get_scanner, ScannerIdEnum, get_field_of_view, FieldOfViewIdEnum
from msotrec.model import DataCollectionModel, AcousticModelWithoutSir

from pipeline import load_sinograms_in_list, process_and_store_sinogram
from volume_generation import create_simple_tissue
from tools import set_logger, plot_npy_two_plots

# FIXME temporary workaround for newest Intel architectures
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Turning off pip env verbosity
PIPENV_VERBOSITY=-1

def run_kwave_forward(
        random_seed: int = 0,
        file_name_root: str = "test_kwave_3D_point_at_focus",
        spacing_mm: int | float = 0.1,
        volume_transducer_dim_mm: int | float = 75,
        volume_planar_dim_mm: int | float = 15,
        volume_height_mm: int | float = 30,
        wavelengths_nm: list = [700],
        simulation_path: str = None
    ) -> None:

    """
    Runs a 3D kWave simulation on a simple pressure distribution with a point at the center of the FOV.
    Only one wavelength, resolution of 100 microns. 

    :param random_seed: seed used for reproducibility
    :param file_name_root: root of the storing file name
    :param spacing_mm: spacing between voxels (in mm). Can't be higher than 0.22 in case of MB reconstruction. 
                        Usually picked in the interval [0.1, 0.2].
    :param volume_transducer_dim_mm: dimension (in mm) of the volume in the horizontal direction of the tranducer plane (x)
    :param volume_planar_dim_mm: dimension (in mm) of the volume in the direction out of the tranducer plane (y)
    :param volume_height_mm: dimension (in mm) of the volume in the vertical direction of the tranducer plane (z) 
                            without taking into account the probe part
    :param wavelengths_nm: list of wavelengths used for simulation
    :param simulation_path: path to the simulation folder
    """

    ### SETTINGS ###
    
    # Definition of the logger to store the log
    volume_name = file_name_root + "_" + str(random_seed)
    output_log_name = "./data/logs/" + volume_name + ".log"
    logger = set_logger(output_log_name)

    # Definition of the recurrent target path
    target_path_hdf5 = simulation_path + "/" + volume_name + ".hdf5"
    target_path_sino = simulation_path + "/" + volume_name

    # Initialize global settings and prepare for simulation pipeline including
    # volume creation and optical forward simulation
    general_settings = {
        Tags.RANDOM_SEED: random_seed,
        Tags.VOLUME_NAME: volume_name,
        Tags.SIMULATION_PATH: simulation_path,
        Tags.SIMPA_OUTPUT_FILE_PATH: target_path_hdf5,
        Tags.SPACING_MM: spacing_mm,
        Tags.DIM_VOLUME_X_MM: volume_transducer_dim_mm,
        Tags.DIM_VOLUME_Y_MM: volume_planar_dim_mm,
        Tags.DIM_VOLUME_Z_MM: volume_height_mm,
        Tags.WAVELENGTHS: wavelengths_nm,
        Tags.GPU: True,
        Tags.US_GEL: False,
        Tags.IGNORE_QA_ASSERTIONS: False, # ignoring a negative pressure problem
        Tags.DO_FILE_COMPRESSION: True,
        Tags.DO_IPASC_EXPORT: True
    }
    settings = sp.Settings(general_settings)

    # Setting specific settings
    settings.set_volume_creation_settings({
        Tags.SIMULATE_DEFORMED_LAYERS: True,
        Tags.STRUCTURES: create_simple_tissue()
    })
    settings.set_optical_settings({
        Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 1e7,
        Tags.OPTICAL_MODEL_BINARY_PATH: os.getenv("MCX_BINARY_PATH"),
        Tags.OPTICAL_MODEL: Tags.OPTICAL_MODEL_MCX,
        Tags.LASER_PULSE_ENERGY_IN_MILLIJOULE: 50
    })
    settings.set_acoustic_settings({
        Tags.ACOUSTIC_SIMULATION_3D: True,
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
        ) 
        # matching that of MB rec, the shift of 8 + 1 mm due to the placement of focus point + height of mediprene membrane
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

    # Running first part of the pipeline to create HDF5 file in which data fields can be added later
    pipeline = [
        sp.ModelBasedAdapter(settings),
        sp.MCXAdapter(settings)
        ]
    sp.simulate(pipeline, settings, device)

    # Dimensions of the whole volume
    # CAUTION : everything needs to be defined relatively to the volume here, and not the FOV.
    # CAUTION : need to go and take them in the dictionary because can be changed by volume adapter function
    number_of_pixels_x = round(settings[Tags.DIM_VOLUME_X_MM] / spacing_mm) 
    number_of_pixels_y = round(settings[Tags.DIM_VOLUME_Y_MM] / spacing_mm)
    number_of_pixels_z = round(settings[Tags.DIM_VOLUME_Z_MM] / spacing_mm)

    # Positioning the points in the FOV
    position_x_point_1 = round(number_of_pixels_x / 2)
    position_z_point_1 = round((43.2 + 8) / spacing_mm)

    # Definition of the initial pressure array
    p0_array = np.zeros((number_of_pixels_x, number_of_pixels_y, number_of_pixels_z))
    p0_array[position_x_point_1, :, position_z_point_1] = np.ones(number_of_pixels_y)
    p0_array = np.array(p0_array, dtype='float64')

    # Saving our new array as the modified initial pressure
    save_data_field(data=p0_array,
                    file_path=target_path_hdf5,
                    data_field=Tags.DATA_FIELD_INITIAL_PRESSURE, 
                    wavelength=wavelengths_nm[0])
    
    # Enabling the simulation to continue without creating a new HDF5 file
    settings[Tags.CONTINUE_SIMULATION] = True

    # Running kwave on the artificial pressure array
    sp.KWaveAdapter(settings).run(device)

    # Loading and storing the obtained pressure sinogram
    sinogram_list = load_sinograms_in_list(file_path=target_path_hdf5, wavelengths=wavelengths_nm)
    
    # Sinogram processing
    resampled_sinogram = process_and_store_sinogram(file_path=target_path_hdf5,
                                                    settings=settings,
                                                    sinogram_list=sinogram_list,
                                                    wavelengths=wavelengths_nm)
    squeezed_sinogram = resampled_sinogram.squeeze(0)

    # Saving the sinogram
    np.save(target_path_sino, squeezed_sinogram)

def run_msot_rec_forward(
        random_seed: int = 0,
        spacing_mm: float | int = 0.1, 
        simulation_path: str = None
    ) -> None:

    """
    Runs the acoustic forward simulation with MSOT rec on a simple pressure distribution with a point at the center of the FOV. 

    :param random_seed: seed used for reproducibility
    :param spacing_mm: spacing between voxels (in mm). Can't be higher than 0.22 in case of MB reconstruction. 
                        Usually picked in the interval [0.1, 0.2].
    :param simulation_path: path to the simulation folder
    """

    # Logging and device infos for the MSOT rec part
    logging.basicConfig(level=logging.INFO)

    # Definition of the target path
    target_path = simulation_path + "/test_msot_rec_2D_point_at_focus_" + str(random_seed)

    # Set input parameters for the model
    scanner_id = ScannerIdEnum.ITHERA_CLINICAL_PTYPE218
    speed_of_sound = 1540
    device = torch.device('cuda:0')
    dtype = torch.float64

    # Dimensions of the FOV
    number_of_pixels_x = 416
    number_of_pixels_z = 416

    # Positioning the points
    position_x_point_1 = round(number_of_pixels_x / 2)
    position_z_point_1 = round((0 + 20.75) / spacing_mm)

    # Creation of the pressure array (in 2D, corresponds to the transducer plane in kWave's 3D array, 
    # i.e. the center in the y direction)
    image = torch.zeros((1, number_of_pixels_x, number_of_pixels_z), device=device, dtype=dtype)
    image[0, position_x_point_1, position_z_point_1] = 1
    image = image.transpose(1, 2) # MSOT rec's FOV shifts x and z directions

    field_of_view = get_field_of_view(field_of_view_id=FieldOfViewIdEnum.ITHERA_CLINICAL_416x416, 
                                      device=device, 
                                      dtype=dtype)

    # Instantiating the model
    scanner = get_scanner(scanner_id=scanner_id, 
                          device=device, 
                          dtype=dtype)

    data_collection_model = DataCollectionModel(scanner=scanner, 
                                                number_of_cropped_samples_at_sinogram_start_1=0, # 5
                                                number_of_cropped_samples_at_sinogram_start_2=0, # 105
                                                number_of_cropped_samples_at_sinogram_end=0,
                                                cutoff_frequency_low=500e3, 
                                                cutoff_frequency_high=12e6, 
                                                windowing_butterworth_order=2,
                                                windowing_minus_3db_point_from_start=300,
                                                windowing_minus_3db_point_from_end=200, 
                                                dtype=dtype, 
                                                device=device)
    
    model = AcousticModelWithoutSir(field_of_view=field_of_view,
                                    data_collection_model=data_collection_model, 
                                    speed_of_sound=speed_of_sound,
                                    device=device, 
                                    dtype=dtype)

    # Performing forward pass
    sinogram = model.forward(image)

    # Sinogram processing
    sinogram = sinogram.squeeze(0)
    sinogram = sinogram.cpu()

    # Storing obtained sinogram
    np.save(target_path, sinogram.numpy())

def compare_msot_rec_and_kwave(
        random_seed: int = 0,
        simulation_path: str = None, 
        target_path: str = None
    ) -> None:

    """
    Compares the acoustic forward models from MSOT rec and kWave (3D) for the reconstruction of one point at the focus of the probe.
    A static offset of 25 frames will be normally be observed.

    :param random_seed: seed used for reproducibility
    :param simulation_path: path to the simulation folder
    :param target_path: path to save the comparison of the two graphs (png)
    """

    # Printing message
    print("Running comparison between kWave and MSOT rec forward acoustic model.")

    run_kwave_forward(random_seed=random_seed, simulation_path=simulation_path)
    run_msot_rec_forward(random_seed=random_seed, simulation_path=simulation_path)

    plot_npy_two_plots(data_path_1=simulation_path + "/test_kwave_3D_point_at_focus_" + str(random_seed) + ".npy",
                       data_path_2=simulation_path + "/test_msot_rec_2D_point_at_focus_" + str(random_seed) + ".npy",
                       label_1='kwave',
                       label_2='msot rec',
                       target_path=target_path)
    
def visualise_device_simpa(
        spacing_mm: int | float = 0.2,
        volume_transducer_dim_mm: int | float = 75,
        volume_planar_dim_mm: int | float = 15,
        volume_height_mm: int | float = 30,
        target_path: str = None
    ) -> None:

    """
    Function to visualise the position of the probe compared to the FOV.

    :param spacing_mm: spacing between voxels
    :param volume_transducer_dim_mm: dimension (in mm) of the volume in the horizontal direction of the tranducer plane (x)
    :param volume_planar_dim_mm: dimension (in mm) of the volume in the direction out of the tranducer plane (y)
    :param volume_height_mm: dimension (in mm) of the volume in the vertical direction of the tranducer plane (z) 
    :param target_path: path to save the image of the visualisation (png)
    """

    # Definition of FOV
    x_min = -20.75
    x_max = 20.75
    z_min = -20.75
    z_max = 20.75
    field_of_view_extent = [x_min, x_max, 0, 0, z_min, z_max] # corresponds to MSOT rec's FOV

    # Definition and positioning of device
    device_position = np.array([volume_transducer_dim_mm/2,
                                volume_planar_dim_mm/2,
                                50])
    device = sp.MSOTAcuityEcho(device_position_mm=device_position)
    device.set_detection_geometry(
        sp.CurvedArrayDetectionGeometry(
            pitch_mm=0.34,
            radius_mm=40,
            number_detector_elements=256,
            detector_element_width_mm=0.24,
            detector_element_length_mm=13,
            center_frequency_hz=3.96e6,
            bandwidth_percent=55,
            sampling_frequency_mhz=40,
            angular_origin_offset=np.pi,
            device_position_mm=device_position,
            field_of_view_extent_mm=field_of_view_extent
            )
        )
    device.add_illumination_geometry(
        sp.SlitIlluminationGeometry()
        )
    
    settings = sp.Settings()
    settings[Tags.DIM_VOLUME_X_MM] = volume_transducer_dim_mm
    settings[Tags.DIM_VOLUME_Y_MM] = volume_planar_dim_mm
    settings[Tags.DIM_VOLUME_Z_MM] = volume_height_mm
    settings[Tags.SPACING_MM] = spacing_mm
    settings[Tags.STRUCTURES] = {}

    positions = device.detection_geometry.get_detector_element_positions_accounting_for_device_position_mm()
    detector_elements = device.detection_geometry.get_detector_element_orientations()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("In volume")
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.quiver(positions[:, 0], positions[:, 2], detector_elements[:, 0], detector_elements[:, 2])
    fov = device.detection_geometry.get_field_of_view_mm()
    plt.plot([fov[0], fov[1], fov[1], fov[0], fov[0]], [fov[4], fov[4], fov[5], fov[5], fov[4]], color="red")
    plt.subplot(1, 2, 2)
    plt.title("Baseline")
    positions = device.detection_geometry.get_detector_element_positions_base_mm()
    fov = device.detection_geometry.field_of_view_extent_mm
    plt.plot([fov[0], fov[1], fov[1], fov[0], fov[0]], [fov[4], fov[4], fov[5], fov[5], fov[4]], color="red")
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.quiver(positions[:, 0], positions[:, 2], detector_elements[:, 0], detector_elements[:, 2])
    plt.tight_layout()

    if target_path is None:
        plt.show()
    else:
        plt.show()
        plt.savefig(target_path)