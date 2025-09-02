import numpy as np
import matplotlib.pyplot as plt

def plot_npy(data_path):

    data = np.load(data_path)
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.show()

def plot_npy_two_plots(data_path_1, data_path_2, label_1, label_2, target_path):

    data_1 = np.load(data_path_1)
    data_2 = np.load(data_path_2)

    # Create a figure with 1 row and 2 columns for the subplots
    _, axs = plt.subplots(2, 1, figsize=(10, 5))  # 1 row, 2 columns

    # Plot the first graph (Sine wave) on the first subplot (axs[0])
    axs[0].imshow(data_1)
    axs[0].set_title(label_1)

    # Plot the second graph (Cosine wave) on the second subplot (axs[1])
    axs[1].imshow(data_2)
    axs[1].set_title(label_2)

    # Adjust the layout to prevent overlap of titles/labels
    plt.tight_layout()

    # Printing the size of the images
    print("Shape first image :", np.shape(data_1))
    print("Shape second image :", np.shape(data_2))

    plt.savefig(target_path, format="png")

    # Show the plot
    plt.show()

def save_png_plot_from_npy(data_path, target_path):

    data = np.load(data_path)
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.savefig(target_path, format="png")