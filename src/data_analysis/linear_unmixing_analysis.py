import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dotenv import load_dotenv
from pathlib import Path
from analysis_tools import crop_image_roi

def violin_plot(
        lu_folder_path : str,
        graph_save_path : str,
        is_roi : bool = False 
    ) -> None:

    """
    Plots a violin plot of the MSE and MAE between LU and GT so2 maps.

    :param lu_folder_path: path to the folder where the LU data is stored
    :param graph_save_path: path to the folder where the violin plot is stored
    :param is_roi: if set to True, the calculated metrics used will be those calculated on the ROI, otherwise on the whole image
    """

    # Retrieving all the files
    files = [os.path.join(lu_folder_path, f) for f in os.listdir(lu_folder_path) if os.path.isfile(os.path.join(lu_folder_path, f))]
    n_images = len(files)

    # Retrieving all the MSEs and MAEs (middle slice)
    mse = []
    mae = []

    for file in files:
        with h5py.File(file, "r") as f:
            if is_roi:
                mse_f = f["middle slice/MSE ROI"][()]
                mae_f = f["middle slice/MAE ROI"][()]
            else:
                mse_f = f["middle slice/MSE"][()]
                mae_f = f["middle slice/MAE"][()]
            mse.append(mse_f)
            mae.append(mae_f)

    # Creation of a dataframe for Seaborn
    data = pd.DataFrame({
        "Error": np.concatenate([mse, mae]),
        "Type": ["MSE"] * n_images + ["MAE"] * n_images
    })

    # Creation of the violin plot
    plt.figure(figsize=(8, 5))
    ax = sns.violinplot(x="Type", y="Error", data=data, inner="box", palette="pastel")

    # Statistical computations (quartiles & std deviation)
    stats = data.groupby("Type")["Error"].agg(["mean", "std", "median", "quantile", "max"])
    stats["Q1"] = data.groupby("Type")["Error"].quantile(0.25)
    stats["Q3"] = data.groupby("Type")["Error"].quantile(0.75)

    # Plotting the stats on the graph
    for i, type_error in enumerate(stats.index):
        mean = stats.loc[type_error, "mean"]
        std = stats.loc[type_error, "std"]
        q1 = stats.loc[type_error, "Q1"]
        q2 = stats.loc[type_error, "median"]
        q3 = stats.loc[type_error, "Q3"]
        max = stats.loc[type_error, "max"]
        
        # Positioning on the graph
        x_pos = np.abs(i - 1)
        y_pos = min(max + 0.1, 0.7)

        # Adding the text
        text = f"mean: {mean:.4f}\nQ1: {q1:.4f}\nQ2: {q2:.4f}\nQ3: {q3:.4f}\nσ: {std:.4f}"
        ax.text(x_pos, y_pos, text, horizontalalignment='center', fontsize=12, color="black", bbox=dict(facecolor='white', alpha=0.6))

    # Ajout de titres et labels
    plt.title("MSE and MAE between GT and LU so2 on 98 images", fontsize=14)
    plt.ylabel("Error", fontsize=12)
    plt.xlabel("Error type", fontsize=12)
    
    # Define limit of y axis
    plt.ylim(0, 1)
        
    plt.savefig(graph_save_path, dpi=300, bbox_inches="tight")
    plt.close()

def linear_regression(
        file_path: str,
        graph_save_path: str,
        roi: tuple = (),
        n_points: int = None
    ) -> None:

    """
    Plots a linear regression between the LU and GT so2 values on one image.

    :param file_path: path to the file where the LU and GT data is stored
    :param graph_save_path: path to the folder where the violin plot is stored
    :param roi: Region Of Interest (ROI) in the form x_min, x_max, z_min, z_max to which the image is cropped for the analysis.
                If set to the empty tuple, no cropping is done.
    :param n_points: number of randomly drawn points to regress on
    """

    # Defining ground truth and unmixed
    with h5py.File(file_path, "r") as f:
        gt = f["middle slice/GT so2"][()]
        lu = f["LU/LU so2"][()]

    # Cropping if roi is specified
    if roi != ():
        x_min, x_max, z_min, z_max = roi
        gt = crop_image_roi(gt, x_min, x_max, z_min, z_max)
        lu = crop_image_roi(lu, x_min, x_max, z_min, z_max)

    # Reshaping data into 1D arrays for regression
    gt_flat = gt.flatten()
    lu_flat = lu.flatten()

    # Sampling the arrays by selecting a certain number of values to analyse, if chosen to
    if n_points != None :

        if n_points > len(gt_flat):
            raise ValueError("Chosen number of points for regression larger than number of pixels on images : {}.".format(len(gt_flat)))
        else:
            indices = np.random.choice(len(gt_flat), size=n_points, replace=False) # randomly sample indices from both arrays

            # Subsampling the arrays based on the selected indices
            gt_flat = gt_flat[indices]
            lu_flat = lu_flat[indices]

    # Performing Linear Regression
    regressor = LinearRegression()
    regressor.fit(gt_flat.reshape(-1, 1), lu_flat) # fitting the model

    # Making predictions based on the regression model
    predictions = regressor.predict(gt_flat.reshape(-1, 1))

    # Plotting the regression results
    plt.figure(figsize=(8, 6))
    ax = plt.scatter(gt_flat, lu_flat, color='blue', alpha=0.5, label="values")
    plt.plot(gt_flat, predictions, color='red', linewidth=2, label="regression")

    # Adding labels and title
    plt.xlabel("Ground truth so2")
    plt.ylabel("LU so2")
    plt.title("Linear Regression between LU and GT so2 pixel values")

    # Displaying the regression equation and metrics
    slope = regressor.coef_[0]
    intercept = regressor.intercept_
    r_squared = regressor.score(gt_flat.reshape(-1, 1), lu_flat)
    mse = mean_squared_error(lu_flat, predictions)
    mae = mean_absolute_error(lu_flat, predictions)
    # regression_equation = "y = {slope:.4f} * x + {intercept:.4f}"

    # Display the regression equation and metrics on the plot
    text_str = f"slope: {slope:.4f}\nintercept: {intercept:.4f}\nR²: {r_squared:.4f}\nMSE: {mse:.4f}\nMAE: {mae:.4f}"
    plt.text(0.1, 0.8, text_str, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3'))

    plt.legend()
    plt.grid(True)
    plt.savefig(graph_save_path)
    plt.close()

def bland_altman(
        file_path: str,
        graph_save_path: str,
        x_axis: str = "mean",
        roi: tuple = (),
        n_points: int = None
    ):

    """
    Plots a Bland Altman plot between, on the one hand, the difference between LU and GT so2 values, and on the other hand the 
    mean between these values on one image.

    :param file_path: path to the file where the Lu and GT data is stored
    :param graph_save_path: path to the folder where the violin plot is stored
    :param x_axis: x axis of the plot (mean, ground truth so2 or LU so2)
    :param roi: Region Of Interest (ROI) in the form x_min, x_max, z_min, z_max to which the image is cropped for the analysis.
                If set to the empty tuple, no cropping is done.
    :param n_points: number of randomly drawn points to regress on
    """

    # Defining ground truth and unmixed
    with h5py.File(file_path, "r") as f:
        gt = f["middle slice/GT so2"][()]
        lu = f["LU/LU so2"][()]

    # Cropping if roi is specified
    if roi != ():
        x_min, x_max, z_min, z_max = roi
        gt = crop_image_roi(gt, x_min, x_max, z_min, z_max)
        lu = crop_image_roi(lu, x_min, x_max, z_min, z_max)

    # Reshaping data into 1D arrays for regression
    gt_flat = gt.flatten()
    lu_flat = lu.flatten()

    # Sampling the arrays by selecting a certain number of values to analyse, if chosen to
    if n_points != None :

        if n_points > len(gt_flat):
            raise ValueError("Chosen number of points for regression larger than number of pixels on images : {}.".format(len(gt_flat)))
        else:
            indices = np.random.choice(len(gt_flat), size=n_points, replace=False) # randomly sample indices from both arrays

            # Subsampling the arrays based on the selected indices
            gt_flat = gt_flat[indices]
            lu_flat = lu_flat[indices]

    # Calculating the differences and the means
    differences = lu_flat - gt_flat
    means = (lu_flat + gt_flat) / 2

    # Calculating the mean difference and standard deviation
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)

    # Choosing the x axis of the plot
    x_axis_list = ["mean", "GT", "LU"]
    if x_axis not in x_axis_list:
        raise ValueError("Chosen x axis not appropriate. Must be in {}.".format(x_axis_list))
    elif x_axis == "mean":
        x = means
        x_label = "Mean of GT and LU"
    elif x_axis == "GT":
        x = gt_flat
        x_label = "GT value"
    elif x_axis == "LU":
        x = lu_flat
        x_label = "LU value"

    # Creating the Bland-Altman plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, differences, color='blue', alpha=0.5, label='differences')
    plt.axhline(mean_diff, color='black', linestyle='--', label=f'mean difference = {mean_diff:.4f}')
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', label=f'+1.96 SD = {mean_diff + 1.96 * std_diff:.4f}')
    plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', label=f'-1.96 SD = {mean_diff - 1.96 * std_diff:.4f}')
    plt.xlabel(x_label)
    plt.ylabel('Difference LU - GT')
    plt.title('Bland-Altman comparison of pixel values in GT and LU so2 maps')

    # Adding a legend and saving the plot
    plt.legend()
    plt.grid(True)
    plt.savefig(graph_save_path)
    plt.close()

if __name__ == "__main__":

    # Loading environment paths
    load_dotenv(Path("path_config.env"))
    data_path = os.getenv("DATA_DIRECTORY")

    violin_plot(lu_folder_path=data_path + "/linear_unmixing/",
                graph_save_path=data_path + "/linear_unmixing_analysis/violin",
                is_roi=True)

    linear_regression(file_path=data_path + "/linear_unmixing/LU_gt_1.hdf5",
                      graph_save_path=data_path + "/linear_unmixing_analysis/LR_gt_1",
                      roi = (107, 308, 250, 300),
                      n_points=1000)

    bland_altman(file_path=data_path + "/linear_unmixing/LU_gt_1.hdf5",
                graph_save_path=data_path + "/linear_unmixing_analysis/bland_altman_gt_1_mean",
                x_axis = "mean",
                roi = (107, 308, 250, 300))