import numpy.random as rd
import simpa as sp

from simpa import Tags
from .volume_creation_tools import add_vessels_to_layers

# Turning off pip env verbosity
PIPENV_VERBOSITY=-1

def create_calf_tissue_gastrocnemius_transversal(
        volume_transducer_dim_mm: int | float = 75,
        volume_planar_dim_mm: int | float = 15,
        volume_height_mm: int | float = 30,
        epsilon_mm: float = 0.2
    ) -> dict:

    """
    Script to generate a 3D calf tissue as close as possible to reality.
    It contains the whole geometrical extent and is scanned on the side of the gastrocnemius with 
    the probe held in the transversal plane.
    Note that the extent doesn't allow to see the large saphenous vein.
    The anterior part was initially more detailed, but was finally simplified into one muscle layer with structures in them. The supplumentary
    level of details can be used to model other geometries.

    :param volume_transducer_dim_mm: dimension of the volume in the x direction (length of the transducer array)
    :param volume_planar_dim_mm: dimension of the volume in the y direction (out-of-imaging-plane direction)
    :param volume_height_mm: dimension of the volume in the z direction (vertical direction, going through tissues)
    :param epsilon_mm: width of the interstitial space between structures

    :returns: the dictionary of anatomical structures to be fed to the model-based creation algorithm
    """

    ### CALCULATION OF ANATOMICAL VALUES ###

    # Value taken at midcalf for average adult.
    # The widest range is taken as a matter of variability (if two sources were found, took union of both ranges)
    widths = {'epidermis': rd.uniform(0.05, 0.15), # rechecked
          'dermis': rd.uniform(1, 4), # rechecked
          'hypodermis': rd.uniform(2, 12), # rechecked
          'gastrocnemius': rd.uniform(10, 30), # rechecked
          'soleus': rd.uniform(15, 35)} # rechecked

    dimensions_x = {'fibularis_longus': rd.uniform(8, 15),
                    'extensor_digitorum_longus': rd.uniform(6, 10),
                    'flexor_digitorum_longus': rd.uniform(10, 15),
                    'tibialis_anterior': rd.uniform(8, 15),
                    'tibialis_posterior': rd.uniform(10, 20)} # x axis in the gastrocnemius transversal case

    dimensions_z = {'fibularis_longus': rd.uniform(8, 15),
                    'extensor_digitorum_longus': rd.uniform(6, 10),
                    'flexor_digitorum_longus': rd.uniform(10, 15),
                    'tibialis_anterior': rd.uniform(8, 15),
                    'tibialis_posterior': rd.uniform(10, 20)}

    radius = {'tibia': rd.uniform(9, 12),
            'fibula': rd.uniform(8, 12),
            'small_saphenous_vein': rd.uniform(0.65, 2.45), 
            # Joh 2013 : mean value of diameter = 3.1 +- 1.3 mm. We took a bit broader (+- 0.5 mm).
            'large_saphenous_vein': rd.uniform(2, 4),
            'anterior_tibial_vein': rd.uniform(0.75, 1.25),
            'posterior_tibial_vein': rd.uniform(1.5, 2),
            'fibular_vein': rd.uniform(1.25, 1.75),
            'anterior_tibial_artery': rd.uniform(1, 1.25),
            'posterior_tibial_artery': rd.uniform(1.5, 2),
            'fibular_artery': rd.uniform(1, 1.5)}

    # Definition of 4 horizontal zones defined by their respective vertical width
    widths['zone_1'] = dimensions_x['fibularis_longus']
    widths['zone_2'] = max(2*radius['fibula'], dimensions_x['extensor_digitorum_longus'])
    widths['zone_3'] = max(2*(radius['fibular_artery'] + radius['fibular_vein'] + radius['posterior_tibial_vein'] + radius['posterior_tibial_artery'] + 3*epsilon_mm),
                 dimensions_x['tibialis_anterior'], dimensions_x['tibialis_posterior'])
    widths['zone_4'] = max(dimensions_x['flexor_digitorum_longus'], 2*radius['tibia'])
    widths['tot_x'] = widths['zone_1'] + widths['zone_2'] + widths['zone_3'] + widths['zone_4'] + 3*epsilon_mm

    # Definition of the tissue depths
    depths = {
        'epidermis': widths['epidermis'],
        'dermis': widths['epidermis'] + widths['dermis'],
        'hypodermis': widths['epidermis'] + widths['dermis'] + widths['hypodermis'],
        'gastrocnemius': widths['epidermis'] + widths['dermis'] + widths['hypodermis'] + epsilon_mm + widths['gastrocnemius'],
        'soleus': widths['epidermis'] + widths['dermis'] + widths['hypodermis'] + epsilon_mm + widths['gastrocnemius'] + epsilon_mm + widths['soleus']
    }

    # Defintion of structure positions in the right leg case
    # CAREFUL : the position of cuboids is defined at the top right angle
    positions_right_leg = {}

    # Zone 1
    positions_right_leg['fibularis_longus'] = [0,
                                                0,
                                                depths['soleus'] + epsilon_mm]
    
    # Zone 2
    positions_right_leg['fibula'] = [widths['zone_1'] + epsilon_mm + widths['zone_2']/2,
                                    0,
                                    depths['soleus'] + epsilon_mm + radius['fibula']]
    positions_right_leg['anterior_tibial_vein'] = [positions_right_leg['fibula'][0],
                                                    0,
                                                    positions_right_leg['fibula'][2] + radius['fibula'] + epsilon_mm + radius['anterior_tibial_vein']]
    positions_right_leg['anterior_tibial_artery'] = [positions_right_leg['fibula'][0],
                                                    0,
                                                    positions_right_leg['anterior_tibial_vein'][2] + radius['anterior_tibial_vein'] + epsilon_mm + radius['anterior_tibial_artery']]
    positions_right_leg['extensor_digitorum_longus'] = [positions_right_leg['fibula'][0] - dimensions_x['extensor_digitorum_longus']/2,
                                                        0,
                                                        positions_right_leg['anterior_tibial_artery'][2] + radius['anterior_tibial_artery'] + epsilon_mm]
    positions_right_leg['small_saphenous_vein'] = [positions_right_leg['fibula'][0],
                                                    0,
                                                    depths['dermis'] + radius['small_saphenous_vein'] +epsilon_mm]
    
    # Zone 3
    positions_right_leg['tibialis_posterior'] = [widths['zone_1'] + epsilon_mm + widths['zone_2'] + epsilon_mm + widths['zone_3']/2 - dimensions_x['tibialis_posterior']/2,
                                                0,
                                                depths['soleus'] + 2*max(radius['fibular_vein'], radius['fibular_artery'], radius['posterior_tibial_vein'], radius['posterior_tibial_artery']) + 2*epsilon_mm]
    positions_right_leg['tibialis_anterior'] = [positions_right_leg['tibialis_posterior'][0] + dimensions_x['tibialis_posterior']/2 - dimensions_x['tibialis_anterior']/2,
                                                0,
                                                positions_right_leg['tibialis_posterior'][2] + dimensions_z['tibialis_posterior'] + epsilon_mm]
    positions_right_leg['fibular_vein'] = [positions_right_leg['tibialis_posterior'][0] + dimensions_x['tibialis_posterior']/2 - epsilon_mm/2 - radius['fibular_vein'],
                                            0,
                                            depths['soleus'] + epsilon_mm + max(radius['fibular_vein'], radius['fibular_artery'], radius['posterior_tibial_vein'], radius['posterior_tibial_artery'])]
    positions_right_leg['fibular_artery'] = [positions_right_leg['fibular_vein'][0] - radius['fibular_vein'] - epsilon_mm - radius['fibular_artery'],
                                            0,
                                            positions_right_leg['fibular_vein'][2]]
    positions_right_leg['posterior_tibial_vein'] = [positions_right_leg['fibular_vein'][0] + radius['fibular_vein'] + epsilon_mm + radius['posterior_tibial_vein'],
                                                    0,
                                                    positions_right_leg['fibular_vein'][2]]
    positions_right_leg['posterior_tibial_artery'] = [positions_right_leg['posterior_tibial_vein'][0] + radius['posterior_tibial_vein'] + epsilon_mm + radius['posterior_tibial_artery'],
                                                    0,
                                                    positions_right_leg['fibular_vein'][2]]
    
    # Zone 4
    positions_right_leg['flexor_digitorum_longus'] = [widths['zone_1'] + epsilon_mm + widths['zone_2'] + epsilon_mm + widths['zone_3'] + epsilon_mm + widths['zone_4']/2 - dimensions_x['flexor_digitorum_longus']/2,
                                                    0,
                                                    depths['soleus'] + epsilon_mm]
    positions_right_leg['tibia'] = [positions_right_leg['flexor_digitorum_longus'][0] + dimensions_x['flexor_digitorum_longus']/2,
                                    0,  
                                    positions_right_leg['flexor_digitorum_longus'][2] + dimensions_z['flexor_digitorum_longus'] + radius['tibia'] + epsilon_mm]

    # Recentering the geometry
    offset_x = (volume_transducer_dim_mm - widths['tot_x'])/2

    positions_right_leg = {element: [position[0] + offset_x,
                                    0,
                                    position[2]]
                        for element, position in positions_right_leg.items()}

    # For the left leg, we mirror all the structures
    positions_left_leg = {element: [volume_transducer_dim_mm - position[0],
                                    0,
                                    position[2]]
                        for element, position in positions_right_leg.items()}
    
    # Some positions have to be recalculated owing to the bad geometries that happen
    positions_left_leg['flexor_digitorum_longus'][0] = positions_left_leg['tibia'][0] - dimensions_x['flexor_digitorum_longus']/2
    positions_left_leg['tibialis_posterior'][0] = positions_left_leg['posterior_tibial_vein'][0] + radius['posterior_tibial_vein'] + epsilon_mm/2 - dimensions_x['tibialis_posterior']/2
    positions_left_leg['tibialis_anterior'][0] = positions_left_leg['tibialis_posterior'][0] + dimensions_x['tibialis_posterior']/2 - dimensions_x['tibialis_anterior']/2
    positions_left_leg['extensor_digitorum_longus'][0] = positions_left_leg['fibula'][0] - dimensions_x['extensor_digitorum_longus']/2
    positions_left_leg['fibularis_longus'][0] = offset_x + widths['zone_4'] + epsilon_mm + widths['zone_3'] + epsilon_mm + widths['zone_2'] + epsilon_mm

    # Random choice of leg laterality (50% chance per side)
    positions = {}

    is_right = rd.choice([True, False])
    if is_right:
        positions = positions_right_leg
    else:
        positions = positions_left_leg

    # Calculating the depth of the anterior zone
    depths['anterior_muscle'] = max(
        depths['soleus'] + epsilon_mm + dimensions_z['fibularis_longus'], 
        positions['extensor_digitorum_longus'][2] + dimensions_z['extensor_digitorum_longus'],
        positions['tibialis_anterior'][2] + dimensions_z['tibialis_anterior'],
        positions['tibia'][2] + radius['tibia']
    ) + epsilon_mm


    ### TISSUE DEFINITIONS ###

    # Background
    background = sp.Settings()
    background[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    background[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND # automatically has priority 0
    
    # Anti-vessel-in-FOV background layer
    anti_vessel_background = sp.Settings()
    anti_vessel_background[Tags.PRIORITY] = 6
    anti_vessel_background[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    anti_vessel_background[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE
    anti_vessel_background[Tags.STRUCTURE_START_MM] = [0, 0, -43.2] # covers the whole probe height
    anti_vessel_background[Tags.STRUCTURE_END_MM] = [0, 0, 0]
    anti_vessel_background[Tags.CONSIDER_PARTIAL_VOLUME] = True
    anti_vessel_background[Tags.ADHERE_TO_DEFORMATION] = True

    # Interstitial tissue
    interstitial_tissue = sp.Settings()
    interstitial_tissue[Tags.PRIORITY] = 1
    interstitial_tissue[Tags.STRUCTURE_START_MM] = [0, 0, depths['hypodermis']] # avoids interstitial tissue to be found over the skin due to its ondulations
    interstitial_tissue[Tags.STRUCTURE_END_MM] = [0, 0, volume_height_mm]
    interstitial_tissue[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.soft_tissue()
    interstitial_tissue[Tags.CONSIDER_PARTIAL_VOLUME] = False
    interstitial_tissue[Tags.ADHERE_TO_DEFORMATION] = True
    interstitial_tissue[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    # Skin
    epidermis = sp.Settings()
    epidermis[Tags.PRIORITY] = 8 # 4 # so that we have epidermis in any case (it is really thin, so can disappear if we give it a lower priority than dermis)
    epidermis[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    epidermis[Tags.STRUCTURE_END_MM] = [0, 0, depths['epidermis']]
    epidermis[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.epidermis(melanin_volume_fraction=rd.uniform(0.01, 0.16)) 
    # [Jacques 1998] :
    # lighted-skinned adults : 1.3 - 6.3 %
    # moderately pigmented adults : 11 - 16 %
    # darkly pigmented adults : 18 - 43 % => not studied for the moment because too low signal
    epidermis[Tags.CONSIDER_PARTIAL_VOLUME] = True
    epidermis[Tags.ADHERE_TO_DEFORMATION] = True
    epidermis[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    dermis = sp.Settings()
    dermis[Tags.PRIORITY] = 3
    dermis[Tags.STRUCTURE_START_MM] = [0, 0, depths['epidermis']]
    dermis[Tags.STRUCTURE_END_MM] = [0, 0, depths['dermis']]
    dermis[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.dermis()
    dermis[Tags.CONSIDER_PARTIAL_VOLUME] = True
    dermis[Tags.ADHERE_TO_DEFORMATION] = True
    dermis[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    hypodermis = sp.Settings()
    hypodermis[Tags.PRIORITY] = 2
    hypodermis[Tags.STRUCTURE_START_MM] = [0, 0, depths['dermis']]
    hypodermis[Tags.STRUCTURE_END_MM] = [0, 0, depths['hypodermis']]
    hypodermis[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.subcutaneous_fat()
    hypodermis[Tags.CONSIDER_PARTIAL_VOLUME] = True
    hypodermis[Tags.ADHERE_TO_DEFORMATION] = True
    hypodermis[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    # Muscles
    # Oxygenation and blood volume fraction values from [Nölke 2024] except for higher bound of BVF that has to be <= 30% to leave 70% water
    gastrocnemius = sp.Settings()
    gastrocnemius[Tags.PRIORITY] = 4 # 5
    gastrocnemius[Tags.STRUCTURE_START_MM] = [0, 0, depths['hypodermis'] + epsilon_mm]
    gastrocnemius[Tags.STRUCTURE_END_MM] = [0, 0, depths['gastrocnemius']]
    gastrocnemius[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.muscle(oxygenation=rd.uniform(0.5, 0.8), blood_volume_fraction=rd.uniform(0.1, 0.3))
    gastrocnemius[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE
    gastrocnemius[Tags.CONSIDER_PARTIAL_VOLUME] = True
    gastrocnemius[Tags.ADHERE_TO_DEFORMATION] = True

    soleus = sp.Settings()
    soleus[Tags.PRIORITY] = 4 # 5
    soleus[Tags.STRUCTURE_START_MM] = [0, 0, depths['gastrocnemius'] + epsilon_mm]
    soleus[Tags.STRUCTURE_END_MM] = [0, 0, depths['soleus']]
    soleus[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.muscle(oxygenation=rd.uniform(0.5, 0.8), blood_volume_fraction=rd.uniform(0.1, 0.3))
    soleus[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE
    soleus[Tags.CONSIDER_PARTIAL_VOLUME] = True
    soleus[Tags.ADHERE_TO_DEFORMATION] = True

    anterior_muscle_layer = sp.Settings()
    anterior_muscle_layer[Tags.PRIORITY] = 4 # 5
    anterior_muscle_layer[Tags.STRUCTURE_START_MM] = [0, 0, depths['soleus'] + epsilon_mm]
    anterior_muscle_layer[Tags.STRUCTURE_END_MM] = [0, 0, depths['anterior_muscle']]
    anterior_muscle_layer[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.muscle(oxygenation=rd.uniform(0.5, 0.8), blood_volume_fraction=rd.uniform(0.1, 0.3))
    anterior_muscle_layer[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE
    anterior_muscle_layer[Tags.CONSIDER_PARTIAL_VOLUME] = True
    anterior_muscle_layer[Tags.ADHERE_TO_DEFORMATION] = True

    # Bones
    tibia = sp.Settings()
    tibia[Tags.PRIORITY] = 7
    tibia[Tags.STRUCTURE_START_MM] = positions['tibia']
    tibia[Tags.STRUCTURE_END_MM] = [positions['tibia'][0], volume_planar_dim_mm, positions['tibia'][2]]
    tibia[Tags.STRUCTURE_RADIUS_MM] = radius['tibia']
    tibia[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.bone()
    tibia[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE
    tibia[Tags.CONSIDER_PARTIAL_VOLUME] = True
    tibia[Tags.ADHERE_TO_DEFORMATION] = False

    fibula = sp.Settings()
    fibula[Tags.PRIORITY] = 7
    fibula[Tags.STRUCTURE_START_MM] = positions['fibula']
    fibula[Tags.STRUCTURE_END_MM] = [positions['fibula'][0], volume_planar_dim_mm, positions['fibula'][2]]
    fibula[Tags.STRUCTURE_RADIUS_MM] = radius['fibula']
    fibula[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.bone()
    fibula[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE
    fibula[Tags.CONSIDER_PARTIAL_VOLUME] = True
    fibula[Tags.ADHERE_TO_DEFORMATION] = False

    # Bigger vessels
    # Oxygenation values from [Nölke 2024]
    small_saphenous_vein = sp.Settings()
    small_saphenous_vein[Tags.PRIORITY] = 5 # 7
    small_saphenous_vein[Tags.STRUCTURE_START_MM] = positions['small_saphenous_vein']
    small_saphenous_vein[Tags.STRUCTURE_DIRECTION] = [0, 1, 0]
    small_saphenous_vein[Tags.STRUCTURE_RADIUS_MM] = radius['small_saphenous_vein']
    small_saphenous_vein[Tags.STRUCTURE_CURVATURE_FACTOR] = 0.005
    small_saphenous_vein[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1
    small_saphenous_vein[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = volume_planar_dim_mm + 1 # won't bifurcate
    small_saphenous_vein[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=rd.uniform(0.6, 0.8))
    small_saphenous_vein[Tags.STRUCTURE_TYPE] = Tags.VESSEL_STRUCTURE
    small_saphenous_vein[Tags.CONSIDER_PARTIAL_VOLUME] = True
    small_saphenous_vein[Tags.ADHERE_TO_DEFORMATION] = True

    anterior_tibial_vein = sp.Settings()
    anterior_tibial_vein[Tags.PRIORITY] = 5 # 7
    anterior_tibial_vein[Tags.STRUCTURE_START_MM] = positions['anterior_tibial_vein']
    anterior_tibial_vein[Tags.STRUCTURE_DIRECTION] = [0, 1, 0]
    anterior_tibial_vein[Tags.STRUCTURE_RADIUS_MM] = radius['anterior_tibial_vein']
    anterior_tibial_vein[Tags.STRUCTURE_CURVATURE_FACTOR] = 0.005
    anterior_tibial_vein[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1
    anterior_tibial_vein[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = volume_planar_dim_mm + 1 # won't bifurcate
    anterior_tibial_vein[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=rd.uniform(0.6, 0.8))
    anterior_tibial_vein[Tags.STRUCTURE_TYPE] = Tags.VESSEL_STRUCTURE
    anterior_tibial_vein[Tags.CONSIDER_PARTIAL_VOLUME] = True
    anterior_tibial_vein[Tags.ADHERE_TO_DEFORMATION] = True

    posterior_tibial_vein = sp.Settings()
    posterior_tibial_vein[Tags.PRIORITY] = 5 # 7
    posterior_tibial_vein[Tags.STRUCTURE_START_MM] = positions['posterior_tibial_vein']
    posterior_tibial_vein[Tags.STRUCTURE_DIRECTION] = [0, 1, 0]
    posterior_tibial_vein[Tags.STRUCTURE_RADIUS_MM] = radius['posterior_tibial_vein']
    posterior_tibial_vein[Tags.STRUCTURE_CURVATURE_FACTOR] = 0.005
    posterior_tibial_vein[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1
    posterior_tibial_vein[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = volume_planar_dim_mm + 1 # won't bifurcate
    posterior_tibial_vein[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=rd.uniform(0.6, 0.8))
    posterior_tibial_vein[Tags.STRUCTURE_TYPE] = Tags.VESSEL_STRUCTURE
    posterior_tibial_vein[Tags.CONSIDER_PARTIAL_VOLUME] = True
    posterior_tibial_vein[Tags.ADHERE_TO_DEFORMATION] = True

    fibular_vein = sp.Settings()
    fibular_vein[Tags.PRIORITY] = 5 # 7
    fibular_vein[Tags.STRUCTURE_START_MM] = positions['fibular_vein']
    fibular_vein[Tags.STRUCTURE_DIRECTION] = [0, 1, 0]
    fibular_vein[Tags.STRUCTURE_RADIUS_MM] = radius['fibular_vein']
    fibular_vein[Tags.STRUCTURE_CURVATURE_FACTOR] = 0.005
    fibular_vein[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1
    fibular_vein[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = volume_planar_dim_mm + 1 # won't bifurcate
    fibular_vein[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=rd.uniform(0.6, 0.8))
    fibular_vein[Tags.STRUCTURE_TYPE] = Tags.VESSEL_STRUCTURE
    fibular_vein[Tags.CONSIDER_PARTIAL_VOLUME] = True
    fibular_vein[Tags.ADHERE_TO_DEFORMATION] = True

    anterior_tibial_artery = sp.Settings()
    anterior_tibial_artery[Tags.PRIORITY] = 5 # 7
    anterior_tibial_artery[Tags.STRUCTURE_START_MM] = positions['anterior_tibial_artery']
    anterior_tibial_artery[Tags.STRUCTURE_DIRECTION] = [0, 1, 0]
    anterior_tibial_artery[Tags.STRUCTURE_RADIUS_MM] = radius['anterior_tibial_artery']
    anterior_tibial_artery[Tags.STRUCTURE_CURVATURE_FACTOR] = 0.005
    anterior_tibial_artery[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1
    anterior_tibial_artery[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = volume_planar_dim_mm + 1 # won't bifurcate
    anterior_tibial_artery[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=rd.uniform(0.9, 1))
    anterior_tibial_artery[Tags.STRUCTURE_TYPE] = Tags.VESSEL_STRUCTURE
    anterior_tibial_artery[Tags.CONSIDER_PARTIAL_VOLUME] = True
    anterior_tibial_artery[Tags.ADHERE_TO_DEFORMATION] = True

    posterior_tibial_artery = sp.Settings()
    posterior_tibial_artery[Tags.PRIORITY] = 5 # 7
    posterior_tibial_artery[Tags.STRUCTURE_START_MM] = positions['posterior_tibial_artery']
    posterior_tibial_artery[Tags.STRUCTURE_DIRECTION] = [0, 1, 0]
    posterior_tibial_artery[Tags.STRUCTURE_RADIUS_MM] = radius['posterior_tibial_artery']
    posterior_tibial_artery[Tags.STRUCTURE_CURVATURE_FACTOR] = 0.005
    posterior_tibial_artery[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1
    posterior_tibial_artery[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = volume_planar_dim_mm + 1 # won't bifurcate
    posterior_tibial_artery[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=rd.uniform(0.9, 1))
    posterior_tibial_artery[Tags.STRUCTURE_TYPE] = Tags.VESSEL_STRUCTURE
    posterior_tibial_artery[Tags.CONSIDER_PARTIAL_VOLUME] = True
    posterior_tibial_artery[Tags.ADHERE_TO_DEFORMATION] = True

    fibular_artery = sp.Settings()
    fibular_artery[Tags.PRIORITY] = 5 # 7
    fibular_artery[Tags.STRUCTURE_START_MM] = positions['fibular_artery']
    fibular_artery[Tags.STRUCTURE_DIRECTION] = [0, 1, 0]
    fibular_artery[Tags.STRUCTURE_RADIUS_MM] = radius['fibular_artery']
    fibular_artery[Tags.STRUCTURE_CURVATURE_FACTOR] = 0.005
    fibular_artery[Tags.STRUCTURE_RADIUS_VARIATION_FACTOR] = 1
    fibular_artery[Tags.STRUCTURE_BIFURCATION_LENGTH_MM] = volume_planar_dim_mm + 1 # won't bifurcate
    fibular_artery[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.blood(oxygenation=rd.uniform(0.9, 1))
    fibular_artery[Tags.STRUCTURE_TYPE] = Tags.VESSEL_STRUCTURE
    fibular_artery[Tags.CONSIDER_PARTIAL_VOLUME] = True
    fibular_artery[Tags.ADHERE_TO_DEFORMATION] = True


    ### TISSUE DICT DEFINITION ###
    
    tissue_dict = sp.Settings()

    # We remove the small saphenous vein in half of the cases because it is avoided in most scanning cases
    is_small_saphenous_there = rd.choice([True, False])
    if is_small_saphenous_there:
        tissue_dict["small saphenous vein"] = small_saphenous_vein

    # The rest of the structures are added
    tissue_dict[Tags.BACKGROUND] = background
    tissue_dict["anti_vessel_background"] = anti_vessel_background
    tissue_dict["interstitial_tissue"] = interstitial_tissue
    tissue_dict["epidermis"] = epidermis
    tissue_dict["dermis"] = dermis
    tissue_dict["hypodermis"] = hypodermis
    tissue_dict["gastrocnemius"] = gastrocnemius
    tissue_dict["soleus"] = soleus
    tissue_dict["anterior_muscle_layer"] = anterior_muscle_layer
    tissue_dict["tibia"] = tibia
    tissue_dict["fibula"] = fibula
    tissue_dict["anterior tibial vein"] = anterior_tibial_vein
    tissue_dict["posterior tibial vein"] = posterior_tibial_vein
    tissue_dict["fibular vein"] = fibular_vein
    tissue_dict["fibular artery"] = fibular_artery
    tissue_dict["anterior tibial artery"] = anterior_tibial_artery
    tissue_dict["posterior tibial artery"] = posterior_tibial_artery

    # Adding vessels to the muscles if specified (little veins and arteries, venules, arterioles)
    (n_veins_hypodermis, n_arteries_hypodermis, n_veins_gastrocnemius, n_arteries_gastrocnemius, n_veins_soleus, 
     n_arteries_soleus, n_veins_anterior_muscle, n_arteries_anterior_muscle) = tuple(rd.randint(20, 30) for _ in range(8))

    tissue_dict = add_vessels_to_layers(n_veins_hypodermis, n_arteries_hypodermis, n_veins_gastrocnemius, 
                                        n_arteries_gastrocnemius, n_veins_soleus, n_arteries_soleus,
                                        n_veins_anterior_muscle, n_arteries_anterior_muscle, widths, 
                                        depths, tissue_dict, volume_transducer_dim_mm, volume_planar_dim_mm,
                                        epsilon_mm)

    return tissue_dict