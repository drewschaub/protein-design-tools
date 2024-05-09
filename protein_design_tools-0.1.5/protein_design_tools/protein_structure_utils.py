import numpy as np
from pathlib import Path
from protein_structure import ProteinStructure

# radius of gyration for alanine helices of different length for all heavy atoms and backbones
ALANINE_HELIX_RADGYR_ALLHEAVY = {11: 5.812949822471507, 12: 6.244455781549667, 13: 6.68504666005375, 14: 7.129365350926259, 15: 7.574489176761235, 16: 8.023545413021195, 17: 8.47638098584884, 18: 8.93003274561916, 19: 9.384838114055489, 20: 9.842240641569205, 21: 10.30096263010548, 22: 10.760037250427935, 23: 11.220361361030104, 24: 11.682115827777798, 25: 12.1442433485779, 26: 12.606810809374334, 27: 13.07041931245057, 28: 13.534628150519183, 29: 13.999006929575993, 30: 14.463936289740502, 31: 14.92954067221723, 32: 15.395348355673045, 33: 15.861326141818935, 34: 16.32785735561372, 35: 16.794718778091966, 36: 17.261640804409602, 37: 17.728868134350854, 38: 18.196467608670215, 39: 18.664179491210756, 40: 19.13200636830553, 41: 19.600105134551917, 42: 20.068410007149975, 43: 20.536767574350847, 44: 21.00528133982524, 45: 21.47402608820051, 46: 21.942852358582567, 47: 22.41173091726601, 48: 22.8807935620039, 49: 23.349987886214556, 50: 23.81922031234207, 51: 24.288542108758794, 52: 24.758015047177793, 53: 25.227538489537572, 54: 25.69710273525735, 55: 26.1667855177001, 56: 26.63655982464055, 57: 27.106348569446638, 58: 27.576215126196107, 59: 28.046166937691716, 60: 28.51617059205183, 61: 28.98619451031134, 62: 29.45629643521247, 63: 29.926464356841024, 64: 30.39664379941636, 65: 30.866870401058417, 66: 31.337167664933176, 67: 31.80750121119959, 68: 32.277838949500556, 69: 32.74824886282797, 70: 33.218697991058875, 71: 33.68916610224486, 72: 34.159663720498116, 73: 34.630211126381234, 74: 35.10078954417609, 75: 35.57137864077206, 76: 36.04200708765933, 77: 36.512671621970746, 78: 36.98334676512827, 79: 37.45404664801668, 80: 37.92478380611943, 81: 38.395546496873166, 82: 38.86631468937534, 83: 39.33711872274857, 84: 39.807947346947174, 85: 40.278783442687775, 86: 40.74963716613754, 87: 41.22051172963705, 88: 41.69141308420812, 89: 42.16232697912556, 90: 42.63325684551918, 91: 43.10421201691082, 92: 43.57516690307307, 93: 44.04613771731921, 94: 44.51713272182241, 95: 44.98813852006201, 96: 45.45915767846676, 97: 45.93019375341659, 98: 46.40124959066299, 99: 46.87231083807635, 100: 47.343375820596385, 101: 47.81446924541347, 102: 48.28556597754189, 103: 48.75666773762817, 104: 49.22778722581732, 105: 49.69892228048183, 106: 50.17006268749256, 107: 50.64120716104254, 108: 51.11237084081527, 109: 51.58354345712508, 110: 52.05471677205773, 111: 52.52590422905195, 112: 52.99710630912558, 113: 53.46830960902495, 114: 53.93951734590935, 115: 54.410740811456186, 116: 54.88196791691665, 117: 55.35320533937211, 118: 55.8244482493513, 119: 56.29570410397095, 120: 56.766959480377054, 121: 57.23823003073362, 122: 57.70950757932899, 123: 58.180791675773214, 124: 58.65207693929248, 125: 59.123372324854934, 126: 59.5946757439899, 127: 60.06597784020172, 128: 60.537289605357714, 129: 61.00860822733064, 130: 61.479929796377846, 131: 61.95125696078086, 132: 62.422590897680344, 133: 62.89393120569489, 134: 63.36527462793193, 135: 63.836622289315244, 136: 64.30797541681409, 137: 64.77933248321732, 138: 65.25069605132184, 139: 65.72205971583705, 140: 66.1934330725741, 141: 66.66480715940588, 142: 67.13618774669605, 143: 67.60757198535957, 144: 68.07896410031367, 145: 68.5503544291711, 146: 69.02175076602694, 147: 69.49315155955986, 148: 69.96455735849433, 149: 70.43596591899521, 150: 70.9073801881394, 151: 71.37879649730824, 152: 71.85021978509693, 153: 72.32163932777677, 154: 72.79306731541374, 155: 73.26450117319462, 156: 73.7359338138831, 157: 74.20737198833012, 158: 74.67880862588255, 159: 75.15025031693378, 160: 75.6216953460464, 161: 76.0931445478091, 162: 76.56459261891496, 163: 77.03604806830619, 164: 77.50750657876385, 165: 77.97896673596793, 166: 78.45042858134278, 167: 78.92189324889077, 168: 79.39336561530286, 169: 79.86483621668933, 170: 80.33630613108387, 171: 80.80778082524652, 172: 81.27925634135704, 173: 81.75073798494533, 174: 82.2222191908206, 175: 82.69370339563758, 176: 83.16518615200998, 177: 83.63667565060963, 178: 84.10816761534626, 179: 84.57966173398079, 180: 85.05115704680117, 181: 85.52265289919676, 182: 85.99415189912996, 183: 86.46565548555583, 184: 86.93715863709824, 185: 87.40866402151529, 186: 87.88017349037693, 187: 88.35168293811121, 188: 88.82319195988562, 189: 89.2947061654878, 190: 89.76622225176006, 191: 90.23773808110077, 192: 90.70925740971637, 193: 91.18077646089289, 194: 91.65229763424811, 195: 92.12381894207486, 196: 92.59534412205021, 197: 93.06686975902831, 198: 93.53839799936588, 199: 94.009924425954, 200: 94.48145693693}
ALANINE_HELIX_RADGYR_BACKBONE = {11: 5.6105073006112764, 12: 6.047199736818816, 13: 6.4917128481079605, 14: 6.939879472066465, 15: 7.389077806935894, 16: 7.841500262012636, 17: 8.297191476394811, 18: 8.753812661492608, 19: 9.21140653034802, 20: 9.671106195012497, 21: 10.132052906689784, 22: 10.593318237143343, 23: 11.055594378190825, 24: 11.519067573178567, 25: 11.982928205119306, 26: 12.44714269997688, 27: 12.91218096690441, 28: 13.37775727387501, 29: 13.843498464961778, 30: 14.309665744687742, 31: 14.776386801200848, 32: 15.243293906369694, 33: 15.7103527999314, 34: 16.177848519563252, 35: 16.64561972630807, 36: 17.113461484463873, 37: 17.58152870329708, 38: 18.04990892009325, 39: 18.518391487139642, 40: 18.986964724562885, 41: 19.45575768362136, 42: 19.924718917429153, 43: 20.393724252994932, 44: 20.862857249795667, 45: 21.332178946523598, 46: 21.801570038670395, 47: 22.2710039713732, 48: 22.74057045849866, 49: 23.210263923292963, 50: 23.679982962513325, 51: 24.14976661638441, 52: 24.619679586986386, 53: 25.089631725580976, 54: 25.559618409771872, 55: 26.029700928536215, 56: 26.499856638260646, 57: 26.970024610338825, 58: 27.440259193234272, 59: 27.91055523177531, 60: 28.380899305698236, 61: 28.851253773923787, 62: 29.321674044271916, 63: 29.79215060970417, 64: 30.262631307391473, 65: 30.73315129284735, 66: 31.20372986933288, 67: 31.674336359300515, 68: 32.14494716052671, 69: 32.61561719958588, 70: 33.08631462488543, 71: 33.55702476586974, 72: 34.02776826327214, 73: 34.498547589513144, 74: 34.96935260536746, 75: 35.44016582440123, 76: 35.911007727512576, 77: 36.38187942588502, 78: 36.852762705588226, 79: 37.323664452174604, 80: 37.794595521899645, 81: 38.26554463627116, 82: 38.7365033453788, 83: 39.207491403602134, 84: 39.678496124128834, 85: 40.149506864723676, 86: 40.620530140706556, 87: 41.09157235374335, 88: 41.56263189356659, 89: 42.03370574351762, 90: 42.50479095182017, 91: 42.97589443825209, 92: 43.447002869586434, 93: 43.9181170598879, 94: 44.389253486939836, 95: 44.86039697769966, 96: 45.33155184333391, 97: 45.8027247537287, 98: 46.27391061622654, 99: 46.745102653712706, 100: 47.21629256542643, 101: 47.68750904498857, 102: 48.15872684445638, 103: 48.6299500067175, 104: 49.101185721188145, 105: 49.57243334667374, 106: 50.043687198751165, 107: 50.5149404739237, 108: 50.98621103537799, 109: 51.45749212659018, 110: 51.92876891862958, 111: 52.40006098780227, 112: 52.87135947945361, 113: 53.342665024852046, 114: 53.81396947199507, 115: 54.28529024799971, 116: 54.75661290097365, 117: 55.22794248304061, 118: 55.69927935025941, 119: 56.17062449349416, 120: 56.641969459519466, 121: 57.11332426465127, 122: 57.58469082347766, 123: 58.05605615319767, 124: 58.527424742694954, 125: 58.998802152012495, 126: 59.47018395907608, 127: 59.94156638046337, 128: 60.41295593216013, 129: 60.88435172257926, 130: 61.3557476604719, 131: 61.827149397741536, 132: 62.29855374432043, 133: 62.76996631822342, 134: 63.24138113250238, 135: 63.71279906183031, 136: 64.18422038950447, 137: 64.65564307622122, 138: 65.12707391583079, 139: 65.59850377958453, 140: 66.06994370829428, 141: 66.54138241982216, 142: 67.01282625041068, 143: 67.48427283095822, 144: 67.95572595820488, 145: 68.42717872702653, 146: 68.89863425356998, 147: 69.37009361064725, 148: 69.84155849922377, 149: 70.3130249106751, 150: 70.78449478690214, 151: 71.25596901493356, 152: 71.72744740284315, 153: 72.19892315432644, 154: 72.67040445892815, 155: 73.14189143781195, 156: 73.61337653329035, 157: 74.08486482344843, 158: 74.55635550705524, 159: 75.02784669517672, 160: 75.49934091050143, 161: 75.97083694658247, 162: 76.44233600498558, 163: 76.91384100986633, 164: 77.38534693946117, 165: 77.85685365494402, 166: 78.3283625477003, 167: 78.79987186941861, 168: 79.27138879340656, 169: 79.74290676607737, 170: 80.21441937347551, 171: 80.68593855769923, 172: 81.15745747398803, 173: 81.62898218921991, 174: 82.1005050761508, 175: 82.57203222277661, 176: 83.04355574974917, 177: 83.5150853262641, 178: 83.98661654650941, 179: 84.45815202400938, 180: 84.92968588243409, 181: 85.40122080024564, 182: 85.87275839752357, 183: 86.34429931953012, 184: 86.81584245709324, 185: 87.28738491023026, 186: 87.7589310039823, 187: 88.23047663364629, 188: 88.70202296286061, 189: 89.17357327912927, 190: 89.64512476530633, 191: 90.11667646634284, 192: 90.5882298202109, 193: 91.05978308555444, 194: 91.53133777956491, 195: 92.0028945397662, 196: 92.47445239448739, 197: 92.9460106326482, 198: 93.4175722188045, 199: 93.88913128926363, 200: 94.36069563660159}

# make a flexible function that returns coordinates from a structure. Function should be able to read all atom coordinates, or cordinates of backbone atoms, or cordinates of CA atoms, or coordinates from specified chains, or from specified residue numbers
def get_coordinates(structure, atom_type="all", chains=None, residue_numbers=None, residue_indices=None, residue_ids=None):
    """
    Get the coordinates of atoms from a ProteinStructure object.

    Parameters:
    structure (ProteinStructure): The protein structure to get the coordinates from.
    atom_type (str, optional): The type of atoms to get the coordinates from. Can be "all", "backbone", or "sidechain". Defaults to "all".
    chains (list of str, optional): A list of chain names to get the coordinates from. If None, get the coordinates from all chains. Defaults to None.
    residue_numbers (list of int, optional): A list of residue numbers to get the coordinates from. If None, get the coordinates from all residues. Defaults to None.
    residue_indices (list of int, optional): A list of residue sequence indices to get the coordinates from. If None, get the coordinates from all residues. Defaults to None.
    residue_ids (list of str, optional): A list of residue sequence identifiers to get the coordinates from. If None, get the coordinates from all residues. Defaults to None.
    
    Returns:
    list of list of float: A list of [x, y, z] coordinates of the selected atoms.
    """

    # Initialize an empty list to store the coordinates
    coordinates = []
    
    # Loop over all chains in the structure
    for chain in structure.chains:
        # Check if the chain is in the specified chains
        if chains is not None and chain.chain_name not in chains:
            continue
        
        # Loop over all residues in the chain
        for residue in chain.residues:
            # Check if the residue number is in the specified residue numbers
            if residue_numbers is not None and residue.res_seq not in residue_numbers:
                continue
            
            # Loop over all atoms in the residue
            for atom in residue.atoms:
                # Check if the atom type is in the specified atom types
                if atom_type == "all" or atom.atom_name == atom_type:
                    # Append the coordinates to the list
                    coordinates.append([atom.x, atom.y, atom.z])
                elif atom_type == "backbone" and atom.atom_name in ["N", "CA", "C", "O"]:
                    coordinates.append([atom.x, atom.y, atom.z])
                elif atom_type == "sidechain" and atom.atom_name not in ["N", "CA", "C", "O"]:
                    coordinates.append([atom.x, atom.y, atom.z])

    # Convert the list of coordinates to a numpy array
    coordinates = np.array(coordinates)
    
    return coordinates

def get_masses(structure, atom_type="all", chains=None, residue_numbers=None, residue_indices=None, residue_ids=None):
    """
    Get the masses of atoms from a ProteinStructure object.

    Parameters:
    structure (ProteinStructure): The protein structure to get the masses from.
    atom_type (str, optional): The type of atoms to get the masses from. Can be "all", "backbone", or "sidechain". Defaults to "all".
    chains (list of str, optional): A list of chain names to get the masses from. If None, get the masses from all chains. Defaults to None.
    residue_numbers (list of int, optional): A list of residue numbers to get the masses from. If None, get the masses from all residues. Defaults to None.
    residue_indices (list of int, optional): A list of residue sequence indices to get the masses from. If None, get the masses from all residues. Defaults to None.
    residue_ids (list of str, optional): A list of residue sequence identifiers to get the masses from. If None, get the masses from all residues. Defaults to None.
    
    Returns:
    list of float: A list of masses of the selected atoms.
    """

    # Initialize an empty list to store the masses
    masses = []
    
    # Loop over all chains in the structure
    for chain in structure.chains:
        # Check if the chain is in the specified chains
        if chains is not None and chain.chain_name not in chains:
            continue
        
        # Loop over all residues in the chain
        for residue in chain.residues:
            # Check if the residue number is in the specified residue numbers
            if residue_numbers is not None and residue.res_seq not in residue_numbers:
                continue
            
            # Loop over all atoms in the residue
            for atom in residue.atoms:
                # Check if the atom type is in the specified atom types
                if atom_type == "all" or atom.atom_name == atom_type:
                    # Append the mass to the list
                    masses.append(atom.mass)
                elif atom_type == "backbone" and atom.atom_name in ["N", "CA", "C", "O"]:
                    masses.append(atom.mass)
                elif atom_type == "sidechain" and atom.atom_name not in ["N", "CA", "C", "O"]:
                    masses.append(atom.mass)

    return masses


def get_radgy(structure, atom_type="backbone", chains=None, residue_numbers=None, residue_indices=None, residue_ids=None):
    """
    Calculate the radius of gyration of a protein structure.

    Parameters:
    structure (ProteinStructure): The protein structure to calculate the radius of gyration for.
    atom_type (str, optional): The type of atoms to calculate the radius of gyration for. Can be "all", "backbone", or "sidechain". Defaults to "backbone".
    chains (list of str, optional): A list of chain names to calculate the radius of gyration for. If None, calculate the radius of gyration for all chains. Defaults to None.
    residue_numbers (list of int, optional): A list of residue numbers to calculate the radius of gyration for. If None, calculate the radius of gyration for all residues. Defaults to None.
    residue_indices (list of int, optional): A list of residue sequence indices to calculate the radius of gyration for. If None, calculate the radius of gyration for all residues. Defaults to None.
    residue_ids (list of str, optional): A list of residue sequence identifiers to calculate the radius of gyration for. If None, calculate the radius of gyration for all residues. Defaults to None.
    
    Returns:
    float: The radius of gyration of the selected atoms.
    """

    # Get the coordinates of all atoms in the structure
    coordinates = get_coordinates(structure, atom_type=atom_type, chains=chains, residue_numbers=residue_numbers, residue_indices=residue_indices, residue_ids=residue_ids)
    masses = get_masses(structure, atom_type=atom_type, chains=chains, residue_numbers=residue_numbers, residue_indices=residue_indices, residue_ids=residue_ids)
    
    # Calculate the center of mass
    center_of_mass = np.average(coordinates, axis=0, weights=masses)

    # Calculate the distance of each atom from the center of mass
    distances = np.linalg.norm(coordinates - center_of_mass, axis=1)

    # Calculate the radius of gyration
    radius_of_gyration = np.sqrt(np.average(distances**2, weights=masses))

    return radius_of_gyration

# get radius of gyration for all alanine helices of different lengths
# structure, atom_type="all", chains=None, residue_numbers=None, residue_indices=None, residue_ids=None):
def get_radgy_alanine_helix(sequence_length, atom_type="backbone"):
    """
    Retrieve the radius of gyration of an ideal alanine helix.

    Parameters:
    sequence_length (int): The length of the alanine helix. The length should be between 11 and 200.
    atom_type (str, optional): The type of atoms to calculate the radius of gyration for. Can be "all" or "backbone". Defaults to "backbone".

    Returns:
    float: The radius of gyration of the alanine helix.
    """
    if atom_type == "all":
        try:
            radius_of_gyration = ALANINE_HELIX_RADGYR_ALLHEAVY[sequence_length]
        except KeyError:
            # Program should break here with error message
            print('Unable to lookup radgyr for the alanine helix. The sequence length must be between 11 and 200.')
            raise
    elif atom_type == "backbone":
        try:
            radius_of_gyration = ALANINE_HELIX_RADGYR_BACKBONE[sequence_length]
        except KeyError:
            # Program should break here with error message
            print('Unable to lookup radgyr for the alanine helix. The sequence length must be between 11 and 200.')
            raise
    else:
        raise ValueError("Invalid atom type. Atom type must be 'all' or 'backbone'.")
    
    return radius_of_gyration

def get_radgy_ratio(structure, atom_type="backbone", chains=None, residue_numbers=None, residue_indices=None, residue_ids=None):

    # Get the radius of gyration
    structure_radius_of_gyration = get_radgy(structure, atom_type=atom_type, chains=chains, residue_numbers=residue_numbers, residue_indices=residue_indices, residue_ids=residue_ids)

    # Get the radius of gyration of an ideal alanine helix of the same length. structure.get_sequences will return the sequendes of each chain of the protein, so these need to be summed
    structure_seq_len = sum([len(sequence) for sequence in structure.get_sequences()])
    alanine_radius_of_gyration = get_radgy_alanine_helix(structure_seq_len, atom_type=atom_type)

    # Calculate the radius of gyration ratio
    radius_of_gyration_ratio = structure_radius_of_gyration / alanine_radius_of_gyration

    return radius_of_gyration_ratio
