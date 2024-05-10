"""
protein_structure.py
====================================
The protein_structure module contains the ProteinStructure class, which represents a protein structure and its 
components.
"""
from pathlib import Path

# Atomic weights from https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii
ATOMIC_WEIGHTS = {
    'H': 1.00794075405578,
    'C': 12.0107358967352,
    'N': 14.0067032114458,
    'O': 15.9994049243183,
    'S': 32.0647874061271
    # Add the rest of the elements here
}

# Dictionary to convert three-letter amino acid codes to one-letter codes
THREE_TO_ONE = {'ALA': 'A',
                'ARG': 'R',
                'ASN': 'N',
                'ASP': 'D',
                'CYS': 'C',
                'GLN': 'Q',
                'GLU': 'E',
                'GLY': 'G',
                'HIS': 'H',
                'ILE': 'I',
                'LEU': 'L',
                'LYS': 'K',
                'MET': 'M',
                'PHE': 'F',
                'PRO': 'P',
                'SER': 'S',
                'THR': 'T',
                'TRP': 'W',
                'TYR': 'Y',
                'VAL': 'V'
                }

class ProteinStructure:
    """ProteinStruture represents a protein structure and its components."""
    def __init__(self, name=None):
        """
        Initialize a ProteinStructure object.

        Parameters
        ----------
        name : str, optional
            The name of the protein structure. Defaults to None.
        """
        self.name = None
        self.chains = []

    class Chain:
        """Chain represents a chain in a protein structure."""
        def __init__(self, name):
            """
            Initialize a Chain object, which will contain a list of Residue objects.
            
            Parameters
            ----------
            name : str
                chainID - chain identifier
            """
            self.name = name
            self.residues = [] # list of Residue objects

        class Residue:
            """Represents a residue in a protein structure."""
            # Residue class will contain a list of Atom objects
            def __init__(self, name, res_seq, i_code):
                """
                Initialize a Residue object, which will contain a list of Atom objects.
                
                Parameters
                ----------
                res_name : str
                    The name of the residue.
                res_seq : int
                    The sequence number of the residue.
                i_code : str
                    The insertion code of the residue.
                """
                self.name = name
                self.res_seq = res_seq
                self.i_code = i_code

                # Unique identifiers for each residue
                self.index = None
                self.res_seq_i = f"{res_seq}{i_code}"
                self.res_name_seq_i = f"{name}{res_seq}{i_code}"

                self.atoms = [] # list of Atom objects

            class Atom:
                """Represents an atom in a protein structure."""
                def __init__(self, atom_id, name, alt_loc, x, y, z, occupancy, temp_factor, segment_id, element, charge):
                    """
                    Initialize an Atom object.
                    
                    Parameters
                    ----------
                    atom_id : int
                        The atom serial number.
                    name : str
                        The atom  name.
                    alt_loc : str
                        The alternate location indicator.
                    x : float
                        The orthogonal coordinates for X in Angstroms.
                    y : float
                        The orthogonal coordinates for Y in Angstroms.
                    z : float
                        The orthogonal coordinates for Z in Angstroms.
                    occupancy : float
                        The occupancy of the atom.
                    temp_factor : float
                        The temperature factor of the atom.
                    segment_id : str
                        The segment ID.
                    element : str
                        The element symbol of the atom.
                    charge : str
                        The charge on the atom.
                    """
                    self.atom_id = atom_id
                    self.name = name
                    self.alt_loc = alt_loc
                    self.x = x
                    self.y = y
                    self.z = z
                    self.occupancy = occupancy
                    self.temp_factor = temp_factor
                    self.segment_id = segment_id
                    self.element = element
                    self.charge = charge

                    # Calculate the mass of the atom
                    self.mass = ATOMIC_WEIGHTS[element]
    
    def read_pdb(self, file_path, chains=None, name=None):
        """
        Read a PDB file and populate the ProteinStructure object with its contents.
        
        Parameters
        ----------
        file_path : str
            The path to the PDB file.
        chains : str or list of str, optional
            The chain(s) to read from the PDB file. If None, read all chains. Defaults to None.
        name : str, optional
            The name of the protein structure. Defaults to None.
        """
        self.name = name # The name of the protein structure. Defaults to None.

        # If the user passed a string for chains, convert it to a list
        if isinstance(chains, str):
            chains = [chains]

        # Parse PDB file and populate self.atoms
        p = Path(file_path)
        if p.suffix == ".pdb":
            with open(p, "r") as f:
                for line in f:
                    if line.startswith("ATOM"):

                        chain_name = line[21].strip()
                        # Check if the chain already exists in self.chains
                        chain = next((c for c in self.chains if c.name == chain_name), None)
                        if chains is None or chain_name in chains:
                            if chain is None:
                                chain = self.Chain(chain_name)
                                self.chains.append(chain)

                            # Check if the residue already exists in self.residues
                            res_name = line[17:20].strip()
                            res_seq = int(line[22:26].strip())
                            i_code = line[26].strip()
                            residue = next((r for r in chain.residues if r.res_seq == res_seq and r.i_code == i_code), None)
                            if residue is None:
                                # Create a Residue object and append it to self.residues
                                residue = self.Chain.Residue(res_name, res_seq, i_code)
                                # Check if the residue index is already assigned, if not assign it
                                if residue.index is None:
                                    # assign it next index value in the chain
                                    residue.index = len(chain.residues)
                                chain.residues.append(residue)

                            # Check if the atom already exists in self.atoms
                            atom_id = int(line[6:11].strip())
                            atom_name = line[12:16].strip()
                            alt_loc = line[16].strip()
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                            occupancy = float(line[54:60].strip())
                            temp_factor = float(line[60:66].strip())
                            segment_id = line[72:76].strip()
                            element = line[76:78].strip()
                            charge = line[78:80].strip()
                            atom = next((a for a in residue.atoms if a.atom_id == atom_id), None)
                            if atom is None:
                                # Create an Atom object and append it to self.atoms
                                atom = self.Chain.Residue.Atom(atom_id, atom_name, alt_loc, x, y, z, occupancy, temp_factor, segment_id, element, charge)
                                residue.atoms.append(atom)

    def get_sequence_dict(self):
        """
        Return the sequence of the protein structure as a dictionary of chains and sequences.

        :returns: A dictionary of chains and sequences. The keys are the chain names and the values are the sequences.
        :rtype: dict
        """
        sequences = {}
        for chain in self.chains:
            sequence = ""
            for residue in chain.residues:
                sequence += THREE_TO_ONE[residue.name]
            sequences[chain.name] = sequence
        return sequences
