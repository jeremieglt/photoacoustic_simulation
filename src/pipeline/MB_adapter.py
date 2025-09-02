import numpy as np
import simpa as sp
import torch
import logging
import skimage

from simpa import Tags
from simpa.io_handling import load_data_field, save_data_field
from msotrec.parametrization import get_scanner, ScannerIdEnum, FieldOfViewIdEnum
from msotrec.model import DataCollectionModel, AcousticModelWithSir, FieldOfView
from msotrec.reconstruction import ModelBasedReconstructor, ScalingForReconstruction
from tools.MB_tools import plot_mb_rec_convergence, plot_mb_rec_l_curve

def adapt_FOV(
        spacing_mm: int | float = 0.1, 
        field_of_view_id: FieldOfViewIdEnum = FieldOfViewIdEnum.ITHERA_CLINICAL_416x416,
        dtype_mb: type = torch.float64,
        torch_device: torch.device = torch.device('cuda:0')
        ) -> FieldOfView:

    """
    Adapting the FOV to correpond to MB reconstruction's requirements. If the spacing is 0.1 mm, then the function will just
    send the FOV corresponding to the FOV id entered, otherwise it will scale it.
    
    :param spacing_mm: spacing between voxels (in mm)
    :param field_of_view_id: MSOT rec id of the FOV wished in MB reconstruction
    :param dtype_mb: dtype used in MB reconstruction
    :param torch_device: device used for torch processing

     :returns: the FOV to be entered for MB reconstruction
    """

    factor = 0.1 / spacing_mm

    if field_of_view_id == FieldOfViewIdEnum.ITHERA_CLINICAL_416x352:
        number_of_pixels_x = round(416 * factor)
        number_of_pixels_z = round(352 * factor)
        x_min = -0.02075
        x_max = 0.02075
        z_min = -0.01435
        z_max = 0.02075

    elif field_of_view_id == FieldOfViewIdEnum.ITHERA_CLINICAL_400x330:
        number_of_pixels_x = round(400 * factor)
        number_of_pixels_z = round(330 * factor)
        x_min = -0.01995
        x_max = 0.01995
        z_min = -0.01295
        z_max = 0.01995

    elif field_of_view_id == FieldOfViewIdEnum.ITHERA_CLINICAL_416x416:
        number_of_pixels_x = round(416 * factor)
        number_of_pixels_z = round(416 * factor)
        x_min = -0.02075
        x_max = 0.02075
        z_min = -0.02075
        z_max = 0.02075

    elif field_of_view_id == FieldOfViewIdEnum.ITHERA_PRECLINICAL_500x500:
        number_of_pixels_x = round(500 * factor)
        number_of_pixels_z = round(500 * factor)
        x_min = -0.012495
        x_max = 0.012495
        z_min = -0.012495
        z_max = 0.012495

    field_of_view = FieldOfView(
        number_of_pixels_x=number_of_pixels_x,
        number_of_pixels_z=number_of_pixels_z,
        x_min=x_min,
        x_max=x_max,
        z_min=z_min,
        z_max=z_max,
        device=torch_device,
        dtype=dtype_mb
        )
    
    return field_of_view

def load_sinograms_in_list(
        file_path: str,
        wavelengths: list = [700, 730, 760, 800, 850, 900]
        ) -> list:

    """
    Loading sinograms at all the simulated wavelengths from SIMPA's storing file.

    :param file_path: path to SIMPA's storing file
    :param wavelengths: list of wavelengths used in the simulation

    :returns: a list of sinograms
    """

    sinogram_list = []
    for wavelength in wavelengths:
        sinogram = sp.load_data_field(str(file_path), Tags.DATA_FIELD_TIME_SERIES_DATA, wavelength)
        sinogram_list.append(sinogram)

    return sinogram_list

def verify_sinogram_dimension(
        sinogram: np.ndarray, 
        nb_sensors_mb: int = 256, 
        nb_time_samples_mb: int = 2030
        ) -> None | ValueError:

    """
    Verifying that the sinogram that we receive as an output from SIMPA has the dimensions suited for processing 
    an input sinogram for MB reconstruction.

    :param sinogram: sinogram outputted by the acoustic simulation
    :param nb_sensors_mb: number of sensors used for the device in MB reconstruction
    :param nb_time_samples_mb: number of time steps used in the sinograms of MB reconstruction

    :returns: nothing if no error is produced, else ValueError
    """
    
    if sinogram.shape[0] != nb_sensors_mb:
        raise ValueError(
                "Number of sensors not matching: {} used in SIMPA. Should be {} to allow for MB reconstruction."
                .format(sinogram.shape[1], nb_sensors_mb)
                )
    
    if sinogram.shape[1] < nb_time_samples_mb:
            raise ValueError(
                "Not enough time steps: {} simulated with SIMPA. Should at least be {} to allow for MB reconstruction. \
                Please increase the spatial dimensions, the voxel spacing or the sampling frequency."
                .format(sinogram.shape[0], nb_time_samples_mb)
                )

def resample_sinogram(
        sinogram: np.ndarray,
        settings: dict,
        sampling_frequency_mhz: int| float = 40,
        nb_time_samples_mb: int = 2030,
        logger = logging.getLogger()
        ) -> np.ndarray:

    """
    Resampling the kWave sinogram so that kWave and MB rec use the same time spacing.

    :param sinogram: sinogram outputted by the acoustic simulation
    :param settings: dictionary of SIMPA settings
    :param sampling_frequency_mhz: sampling frequency used in the PA device (in MHz)
    :param nb_time_samples_mb: number of time steps used in the sinograms of MB reconstruction
    :param logger: logger used in the main pipeline

    :returns: sinogram having the wished number of time samples
    """

    # Dimensions of the kWave sinogram
    nb_detectors, nb_time_samples_k_wave = sinogram.shape

    # Calculating the time step that corresponds to the MB sinogram parameters
    dt_mb = 1 / (sampling_frequency_mhz * 1e6)
    time_of_simu_mb = dt_mb * nb_time_samples_mb

    # Getting the time spacing from kWave simulation, that may be different from MB rec's one because of numerical stability reasons
    dt_kwave = settings['dt_acoustic_sim']

    # Defining the number of time samples that corresponds to the same amount of simulation time in kwave
    # CAUTION: it is lower than the total number of time samples in the kWave sinogram, because kWave simulates wave propagation in the whole volume, 
    # and not only in the FOV
    nb_time_samples_time_of_simu_k_wave = round(time_of_simu_mb / dt_kwave)

    # If this number is different in kWave and MB rec, it means that we have to rescale the sinograms that come out of kWave.
    # If it is not, even if the time step is different, we consider the different neglectable.
    if nb_time_samples_time_of_simu_k_wave != nb_time_samples_mb:
        logger.debug(f"Time step in from kWave acoustic forward model ({dt_kwave}) and MB reconstruction: ({dt_mb}) don't correspond. Setting the time step to the second value.")
        factor = nb_time_samples_mb / nb_time_samples_time_of_simu_k_wave
    else:
        factor = 1

    # Number of time samples in the new kWave sinogram
    nb_time_samples_k_wave_resampled = int(nb_time_samples_k_wave * factor)

    # Resampling
    resampled_sinogram = skimage.transform.resize(
        sinogram, 
        (nb_detectors, nb_time_samples_k_wave_resampled),
        order=5, # max 5
        preserve_range=True
        ) # ne real study on the effect of different resampling

    return resampled_sinogram

def process_and_store_sinogram(
        file_path: str,
        settings: dict,
        sinogram_list: list,
        sampling_frequency_mhz: int | float = 40,
        wavelengths: list = [700, 730, 760, 800, 850, 900],
        nb_sensors_mb: int = 256,
        nb_time_samples_mb: int = 2030,
        dtype_simpa: type = np.float64,
        logger = logging.getLogger()
        ) -> np.ndarray:

    """
    Processing sinograms for MB reconstruction.

    :param file_path: path to the SIMPA storing file
    :param settings: dictionary of SIMPA settings
    :param sinogram_list: list of sinograms obtained after the acoustic simulation in SIMPA (one per wavelength)
    :param sampling_frequency_mhz: sampling frequency used in the PA device (in MHz)
    :param speed_of_sound_m_per_s: speed of sound used during acoustic simulation and reconstruction (in m/s)
    :param wavelengths: list of wavelengths used in the simulation
    :param nb_sensors_mb: number of sensors used for the device in MB reconstruction
    :param nb_time_samples_mb: number of time steps used in the sinograms of MB reconstruction
    :param dtype_simpa: dtype used in the SIMPA sinograms
    :param logger: logger used in the pipeline

    :returns: a stack of sinograms to be fed to the MB reconstruction (needs to be converted to torch tensor before)
    """

    nb_wavelengths = len(wavelengths)
    sinogram_array = np.zeros((nb_wavelengths, nb_sensors_mb, nb_time_samples_mb), dtype=dtype_simpa) # make the sizes of the sinogram match between SIMPA and MSOT rec

    for sinogram, i in zip(sinogram_list, range(nb_wavelengths)):
        
        # Ensuring the the sinograms from kWave can be modified correctly
        verify_sinogram_dimension(sinogram, nb_sensors_mb, nb_time_samples_mb)

        # 1 . Resampling the sinogram in case the time spacing changed in kWave for numerical stability reasons
        sinogram_resampled = resample_sinogram(sinogram, settings, sampling_frequency_mhz, nb_time_samples_mb, logger)

        # 2 . Cropping samples to make kWave and MB rec correspond exactly
        n_cropped_samples = 18 + 25 # 18: these samples are always cropped in MB rec by default, and added up to the ones from the data collection model ; 
                                    # 25: observed static difference between MB rec and kWave => TO BE EXPLAINED
        sinogram_cropped = sinogram_resampled[:, n_cropped_samples: nb_time_samples_mb + n_cropped_samples]

        # Saving the resampled sinogram in the storing file at the corresponding wavelength
        save_data_field(
            sinogram_cropped,
            file_path=file_path,
            data_field=Tags.DATA_FIELD_TIME_SERIES_DATA,
            wavelength=wavelengths[i]
            )
        
        # Filling the final sinogram array with the resampled sinogram at this wavelength
        sinogram_array[i,:,:] = sinogram_cropped

    return sinogram_array

def convert_sinogram(
        sinogram_array: np.ndarray,
        dtype_mb: type = torch.float64,
        torch_device: torch.device = torch.device('cuda:0')
        ) -> torch.tensor:

    """
    Converting a sinogram or stack of sinograms from numpy array to torch tensor.

    :param sinogram_array: sinogram or array of sinograms to be converted
    :param dtype_mb: dtype used in MB reconstruction
    :param torch_device: device used for torch processing

    :returns: a sinogram or stack of sinograms in torch tensor(s)
    """

    sinogram_tensor = torch.tensor(sinogram_array, dtype=dtype_mb, device=torch_device)

    return sinogram_tensor

def resample_pressure(
        file_path: str,
        pressure_array: np.ndarray,
        wavelengths: list = [700, 730, 760, 800, 850, 900]
        ) -> np.ndarray:
    
    """
    Reampling the pressure array sent by MB rec to the dimension wanted in SIMPA.

    :param file_path: path to the SIMPA storing file
    :param settings: dictionary of SIMPA settings
    :param pressure_array: reconstructed pressure array
    :param wavelengths: list of wavelengths used in the simulation

    :returns: resampled pressure array
    """

    nb_wavelengths = len(wavelengths)

    # Loading segmentation and definition of the correct size
    # The segmentation is cropped before reconstrution is started, so it's working. Needs modifications otherwise.
    seg = load_data_field(
        file_path=file_path,
        data_field=Tags.DATA_FIELD_SEGMENTATION,
        wavelength=wavelengths[0]
        )
    
    shape = seg.shape
    if len(shape) == 3:
        (so2_shape_x, so2_shape_z) = shape[0], shape[2] # safer to take this case into account
    elif len(shape) == 2:
        (so2_shape_x, so2_shape_z) = shape[0], shape[1]
    else:
        raise ValueError(
                "Incorrect shape of the segmentation data field. Should be 2 or 3, got: {}".format(len(shape))
                )

    # Resampling
    resampled_pressure_array = skimage.transform.resize(
        pressure_array, 
        output_shape=(nb_wavelengths, so2_shape_x, so2_shape_z), 
        order=5, # max 5
        preserve_range=True
        )

    return resampled_pressure_array

def process_and_store_pressure(
        file_path: str,
        pressure_tensor: torch.tensor,
        wavelengths: list = [700, 730, 760, 800, 850, 900]
        ) -> dict:

    """
    Processing the pressure tensor obtained after reconstruction and save it into the SIMPA storing file.

    :param file_path: path to the SIMPA storing file
    :param pressure_tensor: pressure tensor obtained after MB reconstruction
    :param settings: dictionary of SIMPA settings
    :param wavelengths: list of wavelengths used in the simulation

    :returns: a dictionary of reconstructed pressure with the suited form for storing into the SIMPA storing file
    """

    nb_wavelengths = len(wavelengths)

    pressure_tensor = torch.transpose(pressure_tensor, 1, 2) # reestablishing the shape for SIMPA
    pressure_array = pressure_tensor.cpu().numpy()

    resampled_pressure_array = resample_pressure(file_path, pressure_array, wavelengths)
    
    pressure_dict = {wavelength: resampled_pressure_array[i,:,:] for wavelength, i in zip(wavelengths, range(nb_wavelengths))}

    save_data_field(
        pressure_dict,
        file_path=file_path,
        data_field=Tags.DATA_FIELD_RECONSTRUCTED_DATA
        )

def perform_MB_reconstruction_in_SIMPA(
        file_path: str, 
        volume_name: str, 
        settings: dict,
        field_of_view_id = FieldOfViewIdEnum.ITHERA_CLINICAL_416x416, 
        scanner_id = ScannerIdEnum.ITHERA_CLINICAL_PTYPE218,
        wavelengths: list = [700, 730, 760, 800, 850, 900],
        spacing_mm: float | int = 0.1, 
        sampling_frequency_mhz: float | int = 40,
        speed_of_sound_m_per_s: float | int = 1540, 
        laser_energies: list = [11.252442, 12.537516, 10.592886, 11.040774, 9.934446, 9.74826],
        dtype_simpa: type = np.float64, 
        dtype_mb: type = torch.float64, 
        torch_device: torch.device = torch.device('cuda:0'),
        logger = logging.getLogger(),
        perform_reconstruction: bool = True, 
        plot_convergence: bool = False, 
        plot_l_curve: bool = False
        ) -> None:

    """
    Performing Model-Based reconstruction directly from the SIMPA sinograms. The user can also choose to plot 
    the convergence information or the L-curve of the MB reconstruction.

    :param file_path: path to the SIMPA storing file
    :param volule_name: name of the simulated volume
    :param settings: dictionary of SIMPA settings
    :param field_of_view_id: MSOT rec id of the FOV wished in MB reconstruction
    :param scanner_id: MSOT rec id of the scanner wished in MB reconstruction
    :param spacing_mm: spacing between voxels (in mm)
    :param sampling_frequency_mhz: sampling frequency used in the PA device (in MHz)
    :param speed_of_sound_m_per_s: speed of sound used during acoustic simulation and reconstruction (in m/s)
    :param wavelengths: list of wavelengths used in the simulation
    :param dtype_simpa: dtype used in the SIMPA sinograms
    :param dtype_mb: dtype used in MB reconstruction
    :param torch_device: device used for torch processing
    :param logger: logger used in the main pipeline
    :param perform_reconstruction: if set to True, performs the reconstruction and saves the result in the SIMPA save file.
                                    Can be set to False if the user only wants to plot convergence plot or L-curve.
    :param plot_convergence: if set to True, displays the convergence plot for MB rec
    :param plot_l_curve: if set to True, displays the L-curve for MB rec

    :returns: stores the reconstruction in the SIMPA storing file
    """

    # Defining the FOV
    field_of_view = adapt_FOV(
        spacing_mm=spacing_mm, 
        field_of_view_id=field_of_view_id, 
        dtype_mb=dtype_mb, 
        torch_device=torch_device
        )

    # Defining the scanner device
    scanner = get_scanner(
        scanner_id=scanner_id,
        device=torch_device,
        dtype=dtype_mb
        )
    
    # Defining the data collection model
    data_collection_model = DataCollectionModel(
        scanner=scanner, 
        number_of_cropped_samples_at_sinogram_start_1=0,
        number_of_cropped_samples_at_sinogram_start_2=0, 
        number_of_cropped_samples_at_sinogram_end=0,
        cutoff_frequency_low=500e3, # default value in MSOT rec
        cutoff_frequency_high=12e6, # default value in MSOT rec
        windowing_butterworth_order=2, # default value in MSOT rec
        windowing_minus_3db_point_from_start=300, # default value in MSOT rec
        windowing_minus_3db_point_from_end=200, # default value in MSOT rec
        dtype=dtype_mb, 
        device=torch_device
        )
    
    # Defining the acoustic model
    model = AcousticModelWithSir(
        field_of_view=field_of_view,
        data_collection_model=data_collection_model,
        speed_of_sound=speed_of_sound_m_per_s,
        device=torch_device,
        dtype=dtype_mb
        ) # SIR is modelled in kWave, so it also had to be modelled in MB rec
    
    # Definition of the reconstructor from the raw sinograms (not preprocessed)
    reconstructor = ModelBasedReconstructor(
        model=model, 
        scaling_for_reconstruction=ScalingForReconstruction.L2_NORM_TO_1, 
        regularization_strength=1e-5, # see L-curve plots
        number_of_iterations=50, # max 200
        apply_travel_time_mask=True
        )

    # Loading the storing file and extracting the pressure sinograms
    nb_sensors_mb = scanner.number_of_sensors
    nb_time_samples_mb = scanner.number_of_recorded_time_samples
    sampling_frequency_mhz = sampling_frequency_mhz # [MHz]: CAUTION, if you decide to take here the scanner.sampling_frequency, it's in Hz
    
    sinogram_list = load_sinograms_in_list(file_path, wavelengths)
    
    # Converting them to a suitable format for MSOT rec
    sinogram_array = process_and_store_sinogram(
        file_path=file_path,
        settings=settings,
        sinogram_list=sinogram_list,
        sampling_frequency_mhz=sampling_frequency_mhz,
        wavelengths=wavelengths,
        nb_sensors_mb=nb_sensors_mb,
        nb_time_samples_mb=nb_time_samples_mb,
        dtype_simpa=dtype_simpa,
        logger=logger
        )
    
    sinogram_tensor = convert_sinogram(
        sinogram_array,
        dtype_mb=dtype_mb,
        torch_device=torch_device
        )

    # Plotting a convergence analysis for the reconstruction if asked to (no need to run the reconstruction before, includes it)
    if plot_convergence:
        plot_mb_rec_convergence(
            raw_sinogram_stack=sinogram_tensor,
            laser_energy=laser_energies,
            volume_name=volume_name,
            model=model,
            reconstructor=reconstructor
            )

    # Plotting an L-curve analysis for the reconstruction with the specified regularization strengths if asked to (no need to run the reconstruction before, includes it)
    if plot_l_curve:
        plot_mb_rec_l_curve(
            raw_sinogram_stack=sinogram_tensor,
            laser_energy=laser_energies,
            regularization_strengths=[0.0, 1e-6, 1e-5, 2e-5, 4e-5, 5e-5],
            volume_name=volume_name,
            model=model,
            reconstructor=reconstructor
            )

    if perform_reconstruction:
        # Reconstructing the initial pressure
        pressure_tensor = reconstructor.reconstruct_from_raw(
            raw_sinogram_stack=sinogram_tensor,
            laser_energy_values=laser_energies,
            sensor_indices_for_interpolation={}
            )

        # Storing the result in the SIMPA storing file
        process_and_store_pressure(
            file_path=file_path,
            pressure_tensor=pressure_tensor,
            wavelengths=wavelengths
            )