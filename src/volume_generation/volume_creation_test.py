import simpa as sp

from simpa import Tags

# Turning off pip env verbosity
PIPENV_VERBOSITY=-1

# Setting global params characterizing the simulated volume just in case
VOLUME_TRANSDUCER_DIM_IN_MM = 75
VOLUME_PLANAR_DIM_IN_MM = 15
VOLUME_HEIGHT_IN_MM = 60 # more than in pipeline, to be able to see more tissues than those kept in the simulation volume

def create_simple_tissue() -> dict:

    """
    Example of a very simple tissue with skin layers, one muscle and one vessel. 
    Can be used to get familiarized with the simulation or solve diverse issues.
    """

    background = sp.Settings()
    background[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    background[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    epidermis = sp.Settings()
    epidermis[Tags.PRIORITY] = 4
    epidermis[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    epidermis[Tags.STRUCTURE_END_MM] = [0, 0, 0.1]
    epidermis[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.epidermis(melanin_volume_fraction=0.1)
    epidermis[Tags.CONSIDER_PARTIAL_VOLUME] = True
    epidermis[Tags.ADHERE_TO_DEFORMATION] = True
    epidermis[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    dermis = sp.Settings()
    dermis[Tags.PRIORITY] = 3
    dermis[Tags.STRUCTURE_START_MM] = [0, 0, 0.1]
    dermis[Tags.STRUCTURE_END_MM] = [0, 0, 2]
    dermis[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.dermis()
    dermis[Tags.CONSIDER_PARTIAL_VOLUME] = True
    dermis[Tags.ADHERE_TO_DEFORMATION] = True
    dermis[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    hypodermis = sp.Settings()
    hypodermis[Tags.PRIORITY] = 2
    hypodermis[Tags.STRUCTURE_START_MM] = [0, 0, 2]
    hypodermis[Tags.STRUCTURE_END_MM] = [0, 0, 10]
    hypodermis[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.subcutaneous_fat()
    hypodermis[Tags.CONSIDER_PARTIAL_VOLUME] = True
    hypodermis[Tags.ADHERE_TO_DEFORMATION] = True
    hypodermis[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    muscle = sp.Settings()
    muscle[Tags.PRIORITY] = 1
    muscle[Tags.STRUCTURE_START_MM] = [0, 0, 10.2]
    muscle[Tags.STRUCTURE_END_MM] = [0, 0, 40]
    muscle[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.muscle(oxygenation=0.5, blood_volume_fraction=0.2)
    muscle[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE
    muscle[Tags.CONSIDER_PARTIAL_VOLUME] = True
    muscle[Tags.ADHERE_TO_DEFORMATION] = True

    vessel = sp.Settings()
    vessel[Tags.PRIORITY] = 5
    vessel[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2, 0, 5]
    vessel[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2, VOLUME_PLANAR_DIM_IN_MM, 5]
    vessel[Tags.STRUCTURE_RADIUS_MM] = 1 # [mm]
    vessel[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=0.99)
    vessel[Tags.ADHERE_TO_DEFORMATION] = True
    vessel[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background
    tissue_dict['epidermis'] = epidermis
    tissue_dict['dermis'] = dermis
    tissue_dict['hypodermis'] = hypodermis
    tissue_dict['muscle'] = muscle
    tissue_dict['vessel'] = vessel

    return tissue_dict

def create_vessel_at_focus() -> dict:

    """
    Example of a very simple tissue with one vessel at the probe focus. 
    Can be used to get familiarized with the simulation or solve diverse issues.
    Was used for MB rec integration in the pipeline (adaptation between the two different set-ups).
    """

    background = sp.Settings()
    background[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    background[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    vessel = sp.Settings()
    vessel[Tags.PRIORITY] = 1
    vessel[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2, 0, 8]
    vessel[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2, VOLUME_PLANAR_DIM_IN_MM, 8]
    vessel[Tags.STRUCTURE_RADIUS_MM] = 1 # [mm]
    vessel[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=0.99)
    vessel[Tags.ADHERE_TO_DEFORMATION] = False
    vessel[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE
    vessel[Tags.STRUCTURE_CURVATURE_FACTOR] = 0
    vessel[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 0

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background
    tissue_dict['vessel'] = vessel

    return tissue_dict

def create_3_vessels_around_focus() -> dict:

    """
    Example of a very simple tissue with three vessels around the probe focus. 
    Can be used to get familiarized with the simulation or solve diverse issues.
    Was used for MB rec integration in the pipeline (adaptation between the two different set-ups).
    """

    background = sp.Settings()
    background[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    background[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND

    vessel_1 = sp.Settings()
    vessel_1[Tags.PRIORITY] = 1
    vessel_1[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2, 0, 8]
    vessel_1[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2, VOLUME_PLANAR_DIM_IN_MM, 8]
    vessel_1[Tags.STRUCTURE_RADIUS_MM] = 1 # [mm]
    vessel_1[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=0.99)
    vessel_1[Tags.ADHERE_TO_DEFORMATION] = False
    vessel_1[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_1[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    vessel_2 = sp.Settings()
    vessel_2[Tags.PRIORITY] = 1
    vessel_2[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2 - 15, 0, 10]
    vessel_2[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2, VOLUME_PLANAR_DIM_IN_MM - 15, 10]
    vessel_2[Tags.STRUCTURE_RADIUS_MM] = 1 # [mm]
    vessel_2[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=0.99)
    vessel_2[Tags.ADHERE_TO_DEFORMATION] = False
    vessel_2[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_2[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    vessel_3 = sp.Settings()
    vessel_3[Tags.PRIORITY] = 1
    vessel_3[Tags.STRUCTURE_START_MM] = [2*VOLUME_TRANSDUCER_DIM_IN_MM/2 + 7, 0, 6]
    vessel_3[Tags.STRUCTURE_END_MM] = [2*VOLUME_TRANSDUCER_DIM_IN_MM/2 + 7, VOLUME_PLANAR_DIM_IN_MM, 6]
    vessel_3[Tags.STRUCTURE_RADIUS_MM] = 1 # [mm]
    vessel_3[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=0.99)
    vessel_3[Tags.ADHERE_TO_DEFORMATION] = False
    vessel_3[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel_3[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background
    tissue_dict['vessel_1'] = vessel_1
    tissue_dict['vessel_2'] = vessel_2
    tissue_dict['vessel_3'] = vessel_3

    return tissue_dict