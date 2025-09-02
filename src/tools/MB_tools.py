import matplotlib.pyplot as plt
import torch
from msotrec.model import AcousticModelWithSir
from msotrec.reconstruction import ModelBasedReconstructor

# Turning off pip env verbosity
PIPENV_VERBOSITY=-1

def plot_mb_rec_convergence(
        raw_sinogram_stack : torch.tensor,
        laser_energy : list = [11.252442, 12.537516, 10.592886, 11.040774, 9.934446, 9.74826],
        volume_name : str = "test",
        model = AcousticModelWithSir, 
        reconstructor = ModelBasedReconstructor
    ) -> None:

    """
    Plotting the convergence graph for MB reconstruction.

    :param raw_sinogram_stack: stack of raw sinograms outputted by the acoustic simulation
    :param laser_energy: list of wavelength dependent laser energies
    :param volume_name: name of the simulated volume
    :param model: acoustic model
    :param reconstructor: acoustic reconstructor
    """

    # Preprocessing sinograms
    preprocessed_sinogram = model.data_collection_model.preprocess_sinograms(raw_sinogram_stack=raw_sinogram_stack, 
                            laser_energy_values=laser_energy, sensor_indices_for_interpolation={})

    # Computing convergence analysis
    (image_stacks, data_residual_norms, 
     regularization_terms, objective) = reconstructor.observe_convergence(sinogram_stack=preprocessed_sinogram)

    # Plot convergence analysis
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 8.5))
    ax[0, 0].set_title('Minimization objective')
    ax[0, 1].set_title('Data residual norm')
    ax[0, 2].set_title('Regularization term')
    ax[0, 0].set_yscale('log')
    ax[0, 1].set_yscale('log')
    ax[0, 0].set_xlabel('Number of iterations')
    ax[0, 1].set_xlabel('Number of iterations')
    ax[0, 0].set_xlabel('Number of iterations')

    ax[0, 0].plot(objective.cpu())
    ax[0, 1].plot(data_residual_norms.cpu())
    ax[0, 2].plot(regularization_terms.cpu())

    i1 = int(reconstructor.number_of_iterations / 5)
    p1 = ax[1, 0].imshow(image_stacks[i1, 0, :, :].cpu())
    ax[1, 0].set_title("Iteration {}".format(i1))
    fig.colorbar(p1, ax=ax[1, 0])
    i2 = int(reconstructor.number_of_iterations / 2)
    p2 = ax[1, 1].imshow(image_stacks[i2, 0, :, :].cpu())
    ax[1, 1].set_title("Iteration {}".format(i2))
    fig.colorbar(p2, ax=ax[1, 1])
    i3 = int(reconstructor.number_of_iterations-1)
    p3 = ax[1, 2].imshow(image_stacks[i3, 0, :, :].cpu())
    ax[1, 2].set_title("Iteration {}".format(i3))
    fig.colorbar(p3, ax=ax[1, 2])

    plt.savefig("./data/convergence_analysis_msot_rec/" + volume_name + ".png", dpi=300, bbox_inches='tight')
    plt.show(block=False)

def plot_mb_rec_l_curve(
        raw_sinogram_stack : torch.tensor,
        laser_energy : list = [11.252442, 12.537516, 10.592886, 11.040774, 9.934446, 9.74826],
        regularization_strengths : list = [0.0, 1e-6, 1e-5, 2e-5, 4e-5, 5e-5], 
        volume_name : str = "test",
        model = AcousticModelWithSir, 
        reconstructor = ModelBasedReconstructor
    ) -> None:

    """
    Plotting the L-curve for MB reconstruction.

    :param raw_sinogram_stack: stack of raw sinograms outputted by the acoustic simulation
    :param laser_energy: list of wavelength dependent laser energies
    :param regularization_strengths: list of regularization strengths to be used
    :param volume_name: name of the simulated volume
    :param model: acoustic model
    :param reconstructor: acoustic reconstructor
    """
    
    # Preprocessing sinograms
    preprocessed_sinogram = model.data_collection_model.preprocess_sinograms(raw_sinogram_stack=raw_sinogram_stack, 
                            laser_energy_values=laser_energy, sensor_indices_for_interpolation={})

    # Computing L-curve analysis
    data_residual_norms, regularization_terms, reconstructed_images = reconstructor.compute_l_curve(
        sinogram_stack=preprocessed_sinogram, regularization_strengths=regularization_strengths)
    
    # Visualize L-curve results
    data_residual_norms = data_residual_norms.cpu()
    regularization_terms = regularization_terms.cpu()
    reconstructed_images = reconstructed_images.cpu()

    number_of_regularization_values = reconstructed_images.shape[0]
    _, ax = plt.subplots(nrows=1, ncols=1+number_of_regularization_values, figsize=(20, 3))
    for stack_index in range(data_residual_norms.shape[1]):
        ax[0].plot(
            data_residual_norms[:, stack_index], regularization_terms[:, stack_index], '-o',
            label="image {}".format(stack_index))
        for i_reg_value in range(number_of_regularization_values):
            ax[0].text(
                x=data_residual_norms[i_reg_value, stack_index], y=regularization_terms[i_reg_value, stack_index],
                s=regularization_strengths[i_reg_value])
    ax[0].set_xlabel('Data residual norm')
    ax[0].set_ylabel('Regularization term')
    ax[0].set_title("L-curve")
    ax[0].legend()

    for i in range(number_of_regularization_values):
        ax[1 + i].set_title("$\\lambda$={}".format(regularization_strengths[i]))
        ax[1+i].imshow(reconstructed_images[i, 0, :, :])

    plt.savefig("./data/l_curve_analysis_MSOT_rec/" + volume_name + ".png", dpi=300, bbox_inches='tight')
    plt.show(block=False)