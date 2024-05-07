import numpy as np
from pathlib import Path
from protein_structure import ProteinStructure

# Atomic weights from https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii
ALANINE_HELIX_RADGYR = {11: 5.812949822471507, 12: 6.244455781549667, 13: 6.68504666005375, 14: 7.129365350926259, 15: 7.574489176761235, 16: 8.023545413021195, 17: 8.47638098584884, 18: 8.93003274561916, 19: 9.384838114055489, 20: 9.842240641569205, 21: 10.30096263010548, 22: 10.760037250427935, 23: 11.220361361030104, 24: 11.682115827777798, 25: 12.1442433485779, 26: 12.606810809374334, 27: 13.07041931245057, 28: 13.534628150519183, 29: 13.999006929575993, 30: 14.463936289740502, 31: 14.92954067221723, 32: 15.395348355673045, 33: 15.861326141818935, 34: 16.32785735561372, 35: 16.794718778091966, 36: 17.261640804409602, 37: 17.728868134350854, 38: 18.196467608670215, 39: 18.664179491210756, 40: 19.13200636830553, 41: 19.600105134551917, 42: 20.068410007149975, 43: 20.536767574350847, 44: 21.00528133982524, 45: 21.47402608820051, 46: 21.942852358582567, 47: 22.41173091726601, 48: 22.8807935620039, 49: 23.349987886214556, 50: 23.81922031234207, 51: 24.288542108758794, 52: 24.758015047177793, 53: 25.227538489537572, 54: 25.69710273525735, 55: 26.1667855177001, 56: 26.63655982464055, 57: 27.106348569446638, 58: 27.576215126196107, 59: 28.046166937691716, 60: 28.51617059205183, 61: 28.98619451031134, 62: 29.45629643521247, 63: 29.926464356841024, 64: 30.39664379941636, 65: 30.866870401058417, 66: 31.337167664933176, 67: 31.80750121119959, 68: 32.277838949500556, 69: 32.74824886282797, 70: 33.218697991058875, 71: 33.68916610224486, 72: 34.159663720498116, 73: 34.630211126381234, 74: 35.10078954417609, 75: 35.57137864077206, 76: 36.04200708765933, 77: 36.512671621970746, 78: 36.98334676512827, 79: 37.45404664801668, 80: 37.92478380611943, 81: 38.395546496873166, 82: 38.86631468937534, 83: 39.33711872274857, 84: 39.807947346947174, 85: 40.278783442687775, 86: 40.74963716613754, 87: 41.22051172963705, 88: 41.69141308420812, 89: 42.16232697912556, 90: 42.63325684551918, 91: 43.10421201691082, 92: 43.57516690307307, 93: 44.04613771731921, 94: 44.51713272182241, 95: 44.98813852006201, 96: 45.45915767846676, 97: 45.93019375341659, 98: 46.40124959066299, 99: 46.87231083807635, 100: 47.343375820596385, 101: 47.81446924541347, 102: 48.28556597754189, 103: 48.75666773762817, 104: 49.22778722581732, 105: 49.69892228048183, 106: 50.17006268749256, 107: 50.64120716104254, 108: 51.11237084081527, 109: 51.58354345712508, 110: 52.05471677205773, 111: 52.52590422905195, 112: 52.99710630912558, 113: 53.46830960902495, 114: 53.93951734590935, 115: 54.410740811456186, 116: 54.88196791691665, 117: 55.35320533937211, 118: 55.8244482493513, 119: 56.29570410397095, 120: 56.766959480377054, 121: 57.23823003073362, 122: 57.70950757932899, 123: 58.180791675773214, 124: 58.65207693929248, 125: 59.123372324854934, 126: 59.5946757439899, 127: 60.06597784020172, 128: 60.537289605357714, 129: 61.00860822733064, 130: 61.479929796377846, 131: 61.95125696078086, 132: 62.422590897680344, 133: 62.89393120569489, 134: 63.36527462793193, 135: 63.836622289315244, 136: 64.30797541681409, 137: 64.77933248321732, 138: 65.25069605132184, 139: 65.72205971583705, 140: 66.1934330725741, 141: 66.66480715940588, 142: 67.13618774669605, 143: 67.60757198535957, 144: 68.07896410031367, 145: 68.5503544291711, 146: 69.02175076602694, 147: 69.49315155955986, 148: 69.96455735849433, 149: 70.43596591899521, 150: 70.9073801881394, 151: 71.37879649730824, 152: 71.85021978509693, 153: 72.32163932777677, 154: 72.79306731541374, 155: 73.26450117319462, 156: 73.7359338138831, 157: 74.20737198833012, 158: 74.67880862588255, 159: 75.15025031693378, 160: 75.6216953460464, 161: 76.0931445478091, 162: 76.56459261891496, 163: 77.03604806830619, 164: 77.50750657876385, 165: 77.97896673596793, 166: 78.45042858134278, 167: 78.92189324889077, 168: 79.39336561530286, 169: 79.86483621668933, 170: 80.33630613108387, 171: 80.80778082524652, 172: 81.27925634135704, 173: 81.75073798494533, 174: 82.2222191908206, 175: 82.69370339563758, 176: 83.16518615200998, 177: 83.63667565060963, 178: 84.10816761534626, 179: 84.57966173398079, 180: 85.05115704680117, 181: 85.52265289919676, 182: 85.99415189912996, 183: 86.46565548555583, 184: 86.93715863709824, 185: 87.40866402151529, 186: 87.88017349037693, 187: 88.35168293811121, 188: 88.82319195988562, 189: 89.2947061654878, 190: 89.76622225176006, 191: 90.23773808110077, 192: 90.70925740971637, 193: 91.18077646089289, 194: 91.65229763424811, 195: 92.12381894207486, 196: 92.59534412205021, 197: 93.06686975902831, 198: 93.53839799936588, 199: 94.009924425954, 200: 94.48145693693}

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


def get_radgy(structure, atom_type="all", chains=None, residue_numbers=None, residue_indices=None, residue_ids=None):
    """
    Calculate the radius of gyration of a protein structure.

    Parameters:
    structure (ProteinStructure): The protein structure to calculate the radius of gyration for.
    chains (list of str, optional): A list of chain names to calculate the radius of gyration for. If None, calculate the radius of gyration for all chains. Defaults to None.
    residue_numbers (list of int, optional): A list of residue numbers to calculate the radius of gyration for. If None, calculate the radius of gyration for all residues. Defaults to None.
    residue_indices (list of int, optional): A list of residue sequence indices to calculate the radius of gyration for. If None, calculate the radius of gyration for all residues. Defaults to None.
    residue_ids (list of str, optional): A list of residue sequence identifiers to calculate the radius of gyration for. If None, calculate the radius of gyration for all residues. Defaults to None.
    
    Returns:
    float: The radius of gyration of the selected atoms.
    """

    # Get the coordinates of all atoms in the structure
    coordinates = get_coordinates(structure, atom_type="all", chains=chains, residue_numbers=residue_numbers, residue_indices=residue_indices, residue_ids=residue_ids)
    masses = get_masses(structure, atom_type="all", chains=chains, residue_numbers=residue_numbers, residue_indices=residue_indices, residue_ids=residue_ids)
    
    # Calculate the center of mass
    center_of_mass = np.average(coordinates, axis=0, weights=masses)

    # Calculate the distance of each atom from the center of mass
    distances = np.linalg.norm(coordinates - center_of_mass, axis=1)

    # Calculate the radius of gyration
    radius_of_gyration = np.sqrt(np.average(distances**2, weights=masses))

    return radius_of_gyration

# Test the ProteinStructure class
protein = ProteinStructure()
protein.read_pdb("example.pdb", name="test")

# Print the protein structure
for chain in protein.chains:
    print(f"Chain {chain.chain_name}")
    for residue in chain.residues:
        print(f"Residue {residue.res_name} {residue.res_seq}")
        for atom in residue.atoms:
            print(f"Atom {atom.atom_name} at {atom.x}, {atom.y}, {atom.z} with element {atom.element} has a mass {atom.mass}")

# Get the radius of gyration
radius_of_gyration = get_radgy(protein)
print(f"Radius of gyration: {radius_of_gyration:.8f} Å")

# test this other method and make sure they get the same value
def get_backbone_rg(structure, atom_type="all", chains=None, residue_numbers=None, residue_indices=None, residue_ids=None):
    atom_coords = get_coordinates(structure, atom_type="all", chains=None, residue_numbers=None, residue_indices=None, residue_ids=None)
    atom_masses = get_masses(structure, atom_type="all", chains=None, residue_numbers=None, residue_indices=None, residue_ids=None)

    mass_list = []
    coord_list = []
    for i in range(0, len(atom_coords)):
        x, y, z = atom_coords[i]
        mass = atom_masses[i]
        mass_list.append(mass)
        coord_list.append([x, y, z])

    xm = [(m*i, m*j, m*k) for (i, j, k), m in zip(coord_list, mass_list)]
    tmass = sum(mass_list)
    rr = sum(mi*i + mj*j + mk*k for (i, j, k), (mi, mj, mk) in zip(coord_list, xm))
    mm = sum((sum(i) / tmass)**2 for i in zip(*xm))
    rg = round(math.sqrt(rr / tmass-mm), 6)

    return rg

backbone_rg = get_backbone_rg(protein)
print(f"Backbone Radius of gyration: {backbone_rg:.8f} Å")

alanine_rg_dict = {11: 5.610507, 12: 6.0472, 13: 6.491713, 14: 6.93988, 15: 7.389078, 16: 7.8415, 17: 8.297192, 18: 8.753813, 19: 9.211407, 20: 9.671106, 21: 10.132053, 22: 10.593318, 23: 11.055594, 24: 11.519068, 25: 11.982928, 26: 12.447143, 27: 12.912181, 28: 13.377757, 29: 13.843499, 30: 14.309666, 31: 14.776387, 32: 15.243294, 33: 15.710353, 34: 16.177849, 35: 16.64562, 36: 17.113461, 37: 17.581529, 38: 18.049909, 39: 18.518391, 40: 18.986965, 41: 19.455758, 42: 19.924719, 43: 20.393724, 44: 20.862857, 45: 21.332179, 46: 21.80157, 47: 22.271004, 48: 22.74057, 49: 23.210264, 50: 23.679983, 51: 24.149767, 52: 24.619679, 53: 25.089632, 54: 25.559618, 55: 26.029701, 56: 26.499857, 57: 26.970025, 58: 27.440259, 59: 27.910555, 60: 28.380899, 61: 28.851254, 62: 29.321674, 63: 29.792151, 64: 30.262631, 65: 30.733151, 66: 31.20373, 67: 31.674336, 68: 32.144947, 69: 32.615617, 70: 33.086315, 71: 33.557025, 72: 34.027768, 73: 34.498548, 74: 34.969353, 75: 35.440166, 76: 35.911008, 77: 36.381879, 78: 36.852763, 79: 37.323664, 80: 37.794596, 81: 38.265545, 82: 38.736503, 83: 39.207491, 84: 39.678496, 85: 40.149507, 86: 40.62053, 87: 41.091572, 88: 41.562632, 89: 42.033706, 90: 42.504791, 91: 42.975894, 92: 43.447003, 93: 43.918117, 94: 44.389253, 95: 44.860397, 96: 45.331552, 97: 45.802725, 98: 46.273911, 99: 46.745103, 100: 47.216293, 101: 47.687509, 102: 48.158727, 103: 48.62995, 104: 49.101186, 105: 49.572433, 106: 50.043687, 107: 50.51494, 108: 50.986211, 109: 51.457492, 110: 51.928769, 111: 52.400061, 112: 52.871359, 113: 53.342665, 114: 53.813969, 115: 54.28529, 116: 54.756613, 117: 55.227942, 118: 55.699279, 119: 56.170624, 120: 56.641969, 121: 57.113324, 122: 57.584691, 123: 58.056056, 124: 58.527425, 125: 58.998802, 126: 59.470184, 127: 59.941566, 128: 60.412956, 129: 60.884352, 130: 61.355748, 131: 61.827149, 132: 62.298554, 133: 62.769966, 134: 63.241381, 135: 63.712799, 136: 64.18422, 137: 64.655643, 138: 65.127074, 139: 65.598504, 140: 66.069944, 141: 66.541382, 142: 67.012826, 143: 67.484273, 144: 67.955726, 145: 68.427179, 146: 68.898634, 147: 69.370094, 148: 69.841558, 149: 70.313025, 150: 70.784495, 151: 71.255969, 152: 71.727447, 153: 72.198923, 154: 72.670404, 155: 73.141891, 156: 73.613376, 157: 74.084865, 158: 74.556355, 159: 75.027847, 160: 75.499341, 161: 75.970837, 162: 76.442336, 163: 76.913841, 164: 77.385347, 165: 77.856854, 166: 78.328363, 167: 78.799872, 168: 79.271389, 169: 79.742907, 170: 80.214419, 171: 80.685939, 172: 81.157458, 173: 81.628982, 174: 82.100505, 175: 82.572032, 176: 83.043556, 177: 83.515085, 178: 83.986617, 179: 84.458152, 180: 84.929686, 181: 85.401221, 182: 85.872759, 183: 86.344299, 184: 86.815843, 185: 87.287385, 186: 87.758931, 187: 88.230477, 188: 88.702023, 189: 89.173573, 190: 89.645125, 191: 90.116677, 192: 90.58823, 193: 91.059783, 194: 91.531338, 195: 92.002895, 196: 92.474452, 197: 92.946011, 198: 93.417572, 199: 93.889131, 200: 94.360696}

# The two methods should return the same value
p = Path("/Users/schaubaj/Downloads/alanines")

alanine_rg_dict = {}
# Make a dictionary of the alanine radius of gyration values
for i in range(0, 300):
    ala_path = Path(p, f"{i}_alanines.pdb")
    ala_structure = ProteinStructure()
    try:
        ala_structure.read_pdb(ala_path)
        ala_rg = get_radgy(ala_structure, atom_type="backbone")
        alanine_rg_dict[i+2] = ala_rg

        ala_rg2 = get_backbone_rg(ala_structure, atom_type="backbone")
        print(i+2, ala_rg, ala_rg2)


    except:
        continue

print(alanine_rg_dict)