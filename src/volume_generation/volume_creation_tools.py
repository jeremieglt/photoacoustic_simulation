import numpy.random as rd
import simpa as sp

from simpa import Tags

# Turning off pip env verbosity
PIPENV_VERBOSITY=-1

def add_vessels_to_layers(
        n_veins_hypodermis :int, 
        n_arteries_hypodermis : int, 
        n_veins_gastrocnemius : int, 
        n_arteries_gastrocnemius : int, 
        n_veins_soleus : int, 
        n_arteries_soleus : int, 
        n_veins_anterior_muscle : int, 
        n_arteries_anterior_muscle : int,
        widths_dict : dict, 
        depths_dict : dict, 
        tissue_dict : dict,
        volume_transducer_dim_mm: int | float = 75,
        volume_planar_dim_mm: int | float = 15,
        epsilon_mm: float = 0.2
    ) -> dict:
    
    """
    Script to add vessels in some layers of the created structure. It will add them to the already existing geometry.
    CAREFUL : it only works for the 4 use cases defined.

    :param n_veins_hypodermis: number of veins that you want to generate in the hypodermis
    :param n_arteries_hypodermis: number of arteries that you want to generate in the hypodermis
    :param n_veins_gastrocnemius: number of veins that you want to generate in the gastrocenmius
    :param n_arteries_gastrocnemius: number of arteries that you want to generate in the gastrocenmius
    :param n_veins_soleus: number of veins that you want to generate in the soleus
    :param n_arteries_soleus: number of arteries that you want to generate in the soleus
    :param n_veins_anterior_muscle: number of veins that you want to generate in the anterior muscle
    :param n_arteries_anterior_muscle: number of arteries that you want to generate in the anterior muscle
    :param widths_dict: dictionary containing the widths of all the structures
    :param depths_dict: dictionary containing the depths of all the structures
    :param tissue_dict: dictionary containing all the structures as defined in SIMPA
    :param volume_transducer_dim_mm: dimension of the volume in the x direction (length of the transducer array)
    :param volume_planar_dim_mm: dimension of the volume in the y direction (out-of-imaging-plane direction)
    :param epsilon_mm: width of the interstitial space between structures

    :returns: the dictionary of structures updated with the vessels in layers
    """

    # Adding veins to the hypodermis
    for i in range(1, n_veins_hypodermis + 1):
        tissue_dict["vein_hypodermis_" + str(i)] = create_vessel_in_layer("vein", "hypodermis", 
                                                    widths_dict, depths_dict, volume_transducer_dim_mm, 
                                                    volume_planar_dim_mm, epsilon_mm)
        
    # Adding arteries to the hypodermis
    for i in range(1, n_arteries_hypodermis + 1):
        tissue_dict["artery_hypodermis_" + str(i)] = create_vessel_in_layer("artery", "hypodermis", 
                                                    widths_dict, depths_dict, volume_transducer_dim_mm, 
                                                    volume_planar_dim_mm, epsilon_mm)

    # Adding veins to the gastrocnemius
    for i in range(1, n_veins_gastrocnemius + 1):
        tissue_dict["vein_gastrocnemius_" + str(i)] = create_vessel_in_layer("vein", "gastrocnemius", 
                                                    widths_dict, depths_dict, volume_transducer_dim_mm, 
                                                    volume_planar_dim_mm, epsilon_mm)
        
    # Adding arteries to the gastrocnemius
    for i in range(1, n_arteries_gastrocnemius + 1):
        tissue_dict["artery_gastrocnemius_" + str(i)] = create_vessel_in_layer("artery", "gastrocnemius", 
                                                    widths_dict, depths_dict, volume_transducer_dim_mm, 
                                                    volume_planar_dim_mm, epsilon_mm)
    
    # Adding veins to the soleus
    for i in range(1, n_veins_soleus + 1):
        tissue_dict["vein_soleus_" + str(i)] = create_vessel_in_layer("vein", "soleus", 
                                                    widths_dict, depths_dict, volume_transducer_dim_mm, 
                                                    volume_planar_dim_mm, epsilon_mm)
                                                
    # Adding arteries to the soleus
    for i in range(1, n_arteries_soleus + 1):
        tissue_dict["artery_soleus_" + str(i)] = create_vessel_in_layer("artery", "soleus", 
                                                    widths_dict, depths_dict, volume_transducer_dim_mm, 
                                                    volume_planar_dim_mm, epsilon_mm)

    # Adding veins to the anterior_muscle
    for i in range(1, n_veins_anterior_muscle + 1):
        tissue_dict["vein_anterior_muscle_" + str(i)] = create_vessel_in_layer("vein", "anterior_muscle", 
                                                    widths_dict, depths_dict, volume_transducer_dim_mm, 
                                                    volume_planar_dim_mm, epsilon_mm)
                                                
    # Adding arteries to the anterior_muscle
    for i in range(1, n_arteries_anterior_muscle + 1):
        tissue_dict["artery_anterior_muscle_" + str(i)] = create_vessel_in_layer("artery", "anterior_muscle", 
                                                    widths_dict, depths_dict, volume_transducer_dim_mm, 
                                                    volume_planar_dim_mm, epsilon_mm)
        
    return tissue_dict
        
def create_vessel_in_layer(
        vessel_type : str, 
        layer_name : str,
        widths_dict : dict, 
        depths_dict : dict,
        volume_transducer_dim_mm: int | float = 75,
        volume_planar_dim_mm: int | float = 15,
        epsilon_mm: float = 0.2
    ) -> dict:
    
    """
    Creates a vessel to be added in the chosen layer.

    :param vessel_type: type of vessel chosen (vein or artery)
    :param layer_name: chosen layer (hypodermis, gastrocnemius, soleus or anterior muscle)
    :param widths_dict: dictionary containing the widths of all the structures
    :param depths_dict: dictionary containing the depths of all the structures
    :param volume_transducer_dim_mm: dimension of the volume in the x direction (length of the transducer array)
    :param volume_planar_dim_mm: dimension of the volume in the y direction (out-of-imaging-plane direction)
    :param epsilon_mm: width of the interstitial space between structures

    :return: the vessel dict as defined in SIMPA
    """

    # Choosing and positioning the vessel
    if layer_name == "hypodermis":
        position_z = rd.uniform(depths_dict['dermis'] + epsilon_mm, depths_dict['hypodermis'])
    elif layer_name == "gastrocnemius":
        position_z = rd.uniform(depths_dict['hypodermis'] + epsilon_mm, depths_dict['gastrocnemius'])
    elif layer_name == "soleus":
        position_z = rd.uniform(depths_dict['gastrocnemius'] + epsilon_mm, depths_dict['soleus'])
    else:
        position_z = rd.uniform(depths_dict['soleus'] + epsilon_mm, depths_dict['anterior_muscle'])
    
    position = [
        rd.uniform(0, widths_dict['tot_x']), # starting anywhere in x
        rd.uniform(0, volume_planar_dim_mm), # and anywhere in y
        position_z
        ]
    
    if vessel_type == "vein":
        oxygenation = rd.uniform(0.6, 0.8)
        radius = rd.uniform(0.05, 0.5)
    elif vessel_type == "artery":
        oxygenation = rd.uniform(0.9, 1)
        radius = rd.uniform(0.05, 0.3)

    # Drawing a random direction for the vessel
    direction = list(rd.dirichlet([1, 1, 1])) # generates three numbers summing to 1

    # Definition of the vessel entity
    vessel = sp.Settings()
    vessel[Tags.PRIORITY] = 5 # 7
    vessel[Tags.STRUCTURE_START_MM] = position
    vessel[Tags.STRUCTURE_DIRECTION] = direction
    vessel[Tags.STRUCTURE_RADIUS_MM] = radius
    vessel[Tags.STRUCTURE_CURVATURE_FACTOR] = 0.1
    vessel[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1
    vessel[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = rd.uniform(0, volume_transducer_dim_mm) 
    # will bifurcate somewhere on its length or not
    vessel[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=oxygenation)
    vessel[Tags.STRUCTURE_TYPE] = Tags.VESSEL_STRUCTURE
    vessel[Tags.CONSIDER_PARTIAL_VOLUME] = True
    vessel[Tags.ADHERE_TO_DEFORMATION] = True

    return vessel