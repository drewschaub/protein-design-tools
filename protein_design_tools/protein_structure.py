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

class ProteinStructure:
    # Structure class will contain a list of chain objects
    def __init__(self):
        # Structure objects will contain a list of Chain objects
        self.name = None
        self.chains = []

    class Chain:
        # Chain class will contain a list of residue objects
        def __init__(self, chain_name):
            # Chain objects will contain a list of Residue objects
            self.chain_name = chain_name
            self.residues = []

        class Residue:
            """
            Represents a residue in a protein structure.

            A residue is a specific monomer in the polymeric chain of a protein. Each residue 
            consists of one or more atoms.

            Attributes:
            res_name (str): The name of the residue.
            res_seq (str): The sequence number of the residue.
            i_code (str): The insertion code of the residue.
            index (int): The index of the residue in the chain.
            res_seq_i (str): The unique identifier of the residue.
            atoms (list of Atom): The atoms that make up the residue.
            """
            # Residue class will contain a list of Atom objects
            def __init__(self, res_name, res_seq, i_code):
                # Residue objects will contain a list of Atom objects
                self.res_name = res_name
                self.res_seq = res_seq
                self.i_code = i_code

                # assign a residue index to each residue
                self.index = None                

                # Unique identifier for each residue
                self.res_seq_i = f"{res_seq}{i_code}"
                self.res_name_seq_i = f"{res_name}{res_seq}{i_code}"

                self.atoms = []

            class Atom:
                """
                Represents an atom in a protein structure.

                An atom is a single particle of an element that makes up a residue in a protein structure.

                Attributes:
                atom_id (str): The unique identifier of the atom.
                atom_name (str): The name of the atom.
                alt_loc (str): The alternate location indicator.
                x (float): The x-coordinate of the atom.
                y (float): The y-coordinate of the atom.
                z (float): The z-coordinate of the atom.
                occupancy (str): The occupancy of the atom.
                temp_factor (str): The temperature factor of the atom.
                segment_id (str): The segment identifier of the atom.
                element (str): The element of the atom.
                charge (str): The charge of the atom.
                """
                def __init__(self, atom_id, atom_name, alt_loc, x, y, z, occupancy, temp_factor, segment_id, element, charge):
                    self.atom_id = atom_id
                    self.atom_name = atom_name
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
                    self.mass = ATOMIC_WEIGHTS.get(element, 0.0)
    
    def read_pdb(self, file_path, name=None):
        # Set the name of the protein structure
        self.name = name

        # Parse PDB file and populate self.atoms
        p = Path(file_path)
        if p.suffix == ".pdb":
            with open(p, "r") as f:
                for line in f:
                    if line.startswith("ATOM"):

                        # Check if the chain already exists in self.chains
                        chain_name = line[21].strip()
                        chain = next((c for c in self.chains if c.chain_name == chain_name), None)
                        if chain is None:
                            # Create a Chain object and append it to self.chains
                            chain = self.Chain(chain_name)
                            self.chains.append(chain)

                        # Check if the residue already exists in self.residues
                        res_name = line[17:20].strip()
                        res_seq = line[22:26].strip()
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
                        atom_id = line[6:11].strip()
                        atom_name = line[12:16].strip()
                        alt_loc = line[16].strip()
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        occupancy = line[54:60].strip()
                        temp_factor = line[60:66].strip()
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

        Returns:
        dict: A dictionary of chains and sequences. The keys are the chain names and the values are the sequences.

        """
        sequences = {}
        for chain in self.chains:
            sequence = ""
            for residue in chain.residues:
                sequence += residue.res_name
            sequences[chain.chain_name] = sequence
        return sequences
