"""
protein_structure_utils.py
====================================
The protein_structure_utils module contains functions for analyzing protein structures.
"""
import numpy as np
from pathlib import Path

# make a flexible function that returns coordinates from a structure. Function should be able to read all atom coordinates, or cordinates of backbone atoms, or cordinates of CA atoms, or coordinates from specified chains, or from specified residue numbers
def get_coordinates(structure, atom_type="all", chains=None, residue_numbers=None, residue_indices=None, residue_ids=None):
    """
    Get the coordinates of atoms from a ProteinStructure object.
    
    Parameters
    ----------
    structure : ProteinStructure
        The protein structure to get the coordinates from.
    atom_type : str, optional
        The type of atoms to get the coordinates from. Can be "all", "backbone", or "sidechain". Defaults to "all".
    chains : str or list of str, optional
        A list of chain names to get the coordinates from, or a single chain as a string. If None, get the coordinates 
        from all chains. Defaults to None.
    residue_numbers : list of int, optional
        A list of residue numbers to get the coordinates from. If None, get the coordinates from all residues. Defaults 
        to None.
    residue_indices : list of int, optional
        A list of residue sequence indices to get the coordinates from. If None, get the coordinates from all residues. 
        Defaults to None.
    """
    # If the user passed a string for chains, convert it to a list
    if isinstance(chains, str):
        chains = [chains]

    coordinates = [] # Initialize an empty list to store the coordinates
    
    # Loop over all chains in the structure
    for chain in structure.chains:
        # Check if the chain is in the specified chains
        if chains is not None and chain.name not in chains:
            continue
        
        # Loop over all residues in the chain
        for residue in chain.residues:
            # Check if the residue number is in the specified residue numbers
            if residue_numbers is not None and residue.res_seq not in residue_numbers:
                continue
            
            # Loop over all atoms in the residue append the atom to the coordinates list depending on the atom type
            for atom in residue.atoms:
                if atom_type == "all" or atom.name == atom_type:
                    coordinates.append([atom.x, atom.y, atom.z])
                elif atom_type == "backbone" and atom.name in ["N", "CA", "C", "O"]:
                    coordinates.append([atom.x, atom.y, atom.z])
                elif atom_type == "sidechain" and atom.name not in ["N", "CA", "C", "O"]:
                    coordinates.append([atom.x, atom.y, atom.z])

    coordinates = np.array(coordinates) # Convert the list of coordinates to a numpy array
    
    return coordinates

def get_masses(structure, atom_type="all", chains=None, residue_numbers=None, residue_indices=None, residue_ids=None):
    """
    Get the masses of atoms from a ProteinStructure object.
    
    Parameters
    ----------
    structure : ProteinStructure
        The protein structure to get the masses from.
    atom_type : str, optional
        The type of atoms to get the masses from. Can be "all", "backbone", or "sidechain". Defaults to "all".
    chains : str or list of str, optional
        A list of chain names to get the masses from, or a single chain as a string. If None, get the masses from all 
        chains. Defaults to None.
    residue_numbers : list of int, optional
        A list of residue numbers to get the masses from. If None, get the masses from all residues. Defaults to None.
    residue_indices : list of int, optional
        A list of residue sequence indices to get the masses from. If None, get the masses from all residues. Defaults 
        to None.
    residue_ids : list of str, optional
        A list of residue sequence identifiers to get the masses from. If None, get the masses from all residues. 
        Defaults to None.
    """
    # If the user passed a string for chains, convert it to a list
    if isinstance(chains, str):
        chains = [chains]

    masses = [] # Initialize an empty list to store the masses
    
    # Loop over all chains in the structure
    for chain in structure.chains:
        # Check if the chain is in the specified chains
        if chains is not None and chain.name not in chains:
            continue
        
        # Loop over all residues in the chain
        for residue in chain.residues:
            # Check if the residue number is in the specified residue numbers
            if residue_numbers is not None and residue.res_seq not in residue_numbers:
                continue
            
            # Loop over all atoms in the residue append the atom to the masses list depending on the atom type
            for atom in residue.atoms:
                if atom_type == "all" or atom.name == atom_type:
                    masses.append(atom.mass)
                elif atom_type == "backbone" and atom.name in ["N", "CA", "C", "O"]:
                    masses.append(atom.mass)
                elif atom_type == "sidechain" and atom.name not in ["N", "CA", "C", "O"]:
                    masses.append(atom.mass)

    return masses

def get_radgyr(structure, atom_type="backbone", chains=None, residue_numbers=None, residue_indices=None, residue_ids=None):
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

    # If the user passed a string for chains, convert it to a list
    if isinstance(chains, str):
        chains = [chains]

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
def get_radgyr_alanine_helix(sequence_length, atom_type="backbone"):
    """
    Calculate the radius of gyration of an ideal alanine helix. The radius of gyration is calculated from a linear 
    regression model performed on a dataset of ideal alanine helices of different lengths.
    
    Parameters
    ----------
    sequence_length : int
        The length of the alanine helix.
    atom_type : str, optional
        The type of atoms to calculate the radius of gyration for. Can be "all" or "backbone". Defaults to "backbone".
    """
    if atom_type == "all":
        # values are from a linear regression of a dataset of ideal alanine helices of different lengths
        radius_of_gyration = 0.4704427974768968 * float(sequence_length) + 0.33788445679512336
    elif atom_type == "backbone":
        # values are from a linear regression of a dataset of ideal alanine helices of different lengths
        radius_of_gyration = 0.4706800060715209 * float(sequence_length) + 0.17986881616526773
    else:
        raise ValueError("Invalid atom type. Atom type must be 'all' or 'backbone'.")
    
    return radius_of_gyration

def get_radgyr_ratio(structure, atom_type="backbone", chains=None):
    """
    Calculate the radius of gyration ratio of a protein structure to an ideal alanine helix of the same length.
    
    Parameters
    ----------
    structure : ProteinStructure
        The protein structure to calculate the radius of gyration ratio for.
    atom_type : str, optional
        The type of atoms to calculate the radius of gyration for. Can be "all" or "backbone". Defaults to "backbone".
    chains : list of str, optional
        A list of chain names to calculate the radius of gyration for. If None, calculate the radius of gyration for 
        all chains. Defaults to None.
    """

    # If the user passed a string for chains, convert it to a list
    if isinstance(chains, str):
        chains = [chains]

    # Get the radius of gyration
    structure_radius_of_gyration = get_radgyr(structure, atom_type=atom_type, chains=chains)

    # get the sequence length of the protein
    alanine_helix_length = 0
    sequence_dict = structure.get_sequence_dict()
    for chain in sequence_dict.keys():
        if chains is None or chain in chains:
            alanine_helix_length += len(sequence_dict[chain])
        else:
            continue

    alanine_radius_of_gyration = get_radgyr_alanine_helix(alanine_helix_length, atom_type=atom_type)

    # Calculate the radius of gyration ratio
    radius_of_gyration_ratio = structure_radius_of_gyration / alanine_radius_of_gyration

    return radius_of_gyration_ratio
