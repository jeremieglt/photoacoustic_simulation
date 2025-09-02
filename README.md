# Photoacoustic simulation

This package was developed in the scope of a master's thesis project (Jérémie Gillet, MSc Mechanical Engineering, PDF included in the repo). It provides the possibility to simulate photoacoustic imaging from end-to-end, including model-based anatomical volume creation, optical forward modeling, acoustic forward modeling and acoustic reconstruction, and focuses on gastrocnemius muscle scanning in the transverse plane for PAD diagnosis and follow-up. The final aim in this case was to generate a diverse dataset to train Deep Learning models for spectral unmixing in the use case previously defined.

The repo is mostly based on the SIMPA toolkit by the DKFZ (German Center for Cancer Research). See paper [here](https://doi.org/10.1117/1.JBO.27.8.083010). What was mostly modified/added here:
* model-based generation of calf tissue mimicking geometries with structural and physiological variability, with emphasis on realism
* adaptation to the usual experimental set-up for PAD photoacoustic scanning (Acuity Echo device) 
* addition of MSOT rec for more accurate reconstruction
* shaping of a data generation pipeline, including training data preprocessing (data selection, drawing of ROIs)

## Setup

The simulation package has to be installed manually from the GitHub repository: 
1. `git clone https://github.com/jeremieglt/photoacoustic_simulation`
2. `cd photoacoustic_simulation`
3. `git checkout main`
4. `git pull`

A Pipfile is provided with all the required dependencies. It is advised to create a specific virtual environment, either by creating a .venv folder here, or without doing it. The following command:

    pipenv install

will install the dependencies in .venv if created or in `C:\Users\user_name\.virtualenvs` if not.

## External tools

These third party toolkits are required for the simulations:

### SIMPA

The structure of the pipeline follows the classical sketch from SIMPA. Some practical information were gathered by them in the [understanding SIMPA documentation](./docs/source/understanding_simpa.md) that was added to this repos in case the user would need more details. For any unfound information, check the [official documentation](https://simpa.readthedocs.io/en/develop).

For the means of our simulations, a few add-ons were made to SIMPA, and pull requests were made as a consequence: 
* addition of wavelength-dependent laser energy (T417)
* addition of a 3D virtual twin for Acuity Echo to allow for 3D acoustic simulation (STILL TO BE DONE)

A custom version of SIMPA hosted under the iThera Azure and including these add-ons has to be used until these functionalities are eventually added by DKFZ. It is loaded in the Pipfile.

### mcx (Optical Forward Model)

Download the latest nightly build of [mcx](http://mcx.space/) on [this page](http://mcx.space/nightly/github/) for your operating system:

- Linux: `mcx-linux-x64-github-latest.zip`
- MacOS: `mcx-macos-x64-github-latest.zip`
- Windows: `mcx-windows-x64-github-latest.zip`

Then extract the files and set `MCX_BINARY_PATH=/.../mcx/bin/mcx` in your path_config.env.

### k-Wave (Acoustic Forward Model)

Please follow the following steps and use the k-Wave install instructions 
for further (and much better) guidance under:

[http://www.k-wave.org/](http://www.k-wave.org/)

1. Install MATLAB with the core, image processing and parallel computing toolboxes activated at the minimum.
2. Download the kWave toolbox (version >= 1.4)
3. Add the kWave toolbox base path to the toolbox paths in MATLAB
4. If wanted: Download the CPP and CUDA binary files and place them in the k-Wave/binaries folder
5. Note down the system path to the `matlab` executable file.

## Path management

To ensure that the codes run properly, the paths defined in `path_config.env` have to be redefined relatively to the configuration that you use on your local hard drive. We then use the `os` package to refer to these paths in the code. Other methods can be used if you prefer, but this requires adaptations in the code. We did not use the SIMPA `PathManager`.

## Main use

The main files that you might want to run are in the `src\` root:
* `pipeline.py`: runs a single pipeline (optical + acoustic forward, reconstruction) with the chosen configuration and stores the HDF5 data in `DATA_DIRECTORY`. One complete pipeline (6 wavelengths) takes approximately 1 h 40 min to run.
* `generate_raw_data.py`: runs a defined number of pipelines on randomly initialized anatomical volumes and stores the HDF5 data in the chosen `RAW_DATA_SAVE_DIRECTORY`. Approximately 100 simulations can be computed per week.
* `generate_selected_data.py`: selects only the necessary fields from the raw data and stores the HDF5 files in `SELECTED_DATA_SAVE_DIRECTORY`
* `generate_rois.py`: allows to select manually ROIs on the selected data and stores them in the initial file
* `tests.py`: allows to perform diverse tests defined in the `tests\` folder

These files have to be ran from there to allow for folders to interact properly.

## Volume generation

Multiple geometries mimicking anatomy were generated by the model-based algorithm from SIMPA, allowing to arrange simple geometrical shapes in space and allocate them the wanted physical properties:
* `volume_creation_test.py`: simple tissues to perform tests, e.g. placing the probe correctly in SIMPA or ensuring compatibility between the acoustic forward model (SIMPA) and acoustic reconstruction (MSOT rec)
* `volume_creation_calf.py`: calf-mimicking tissues to model a physiological calf geometry scanned in the transverse plane from the gastrocnemius muscle side. Variability in the data comes from the variation of geometrical (radius of veins/bones, widths of vertical layers) and physical parameters (oxygenation, water or melanin content). Similar scripts could be written to simulate other geometries. In particular, scanning the tibialis anterior in the sagittal plane is common in the clinical practice, and this scenario is easily adaptable from the one we just described by transposing the volume in the correct directions.

In order to visualize the created volume, running the main pipeline by keeping only the volume generation block with the selected volume is the way to go. The HDF5 file will contain the whole 3D segmentation.

## Data preprocessing

The `data_preprocessing\` folder mostly contains tools for data (`generate_selected_data.py`) and roi (`generate_rois.py`) selection.

## Data analysis

In the `data_analysis\` folder, you will find a few scripts to analyze the generated data:
* `linear_unmixing.py`: performing non-negative Linear Unmixing (LU) on the reconstructed pressure images
* `linear_unmixing_analysis.py`: analyzing the LU results (ccomparison to ground truth, ...)
* `reconstruction_analysis.py`: analyzing the reconstructed pressure data (comparison to ideal pressure image, plot of pixel-wise spectrum)

## Contacts

Jérémie Gillet (jeremie.gillet@gmail.com) \
Guillaume Zahnd (guillaume.zahnd@ithera-medical.com)