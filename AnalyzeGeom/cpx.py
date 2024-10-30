import numpy as np
import pandas as pd
import os
from ..globalvars import ATOMIC_NUM_TO_ELEMENT, COVALENT_RADII, TRANSITION_METALS, ELEMENT_TO_ATOMIC_NUM

def remove_digits(element_with_numbering):
    # Iterate through the string and keep only non-digit characters
    element = ''.join([char for char in element_with_numbering if not char.isdigit()])
    return element

def normalize(vector):
    """ Normalize a vector. """
    return vector / np.linalg.norm(vector)

class Atom:
    def __init__(self, idx, sym, coord):
        self.idx = idx
        self.sym = sym
        self.coord_x = coord[0]
        self.coord_y = coord[1]
        self.coord_z = coord[2]
        self.coord = np.array(coord)
        
    def __repr__(self):
        return f"Atom({self.sym}{self.idx})"
    
    @property
    def atomic_num(self):
        return ELEMENT_TO_ATOMIC_NUM.get(self.sym, np.nan)
    
    @property
    def cov_radii(self):
        return COVALENT_RADII.get(self.sym, 0.0)
    
    def distance_to(self, other):
        return np.linalg.norm(self.coord - other.coord)

class Mol:
    def __init__(self):
        self.atoms = []
        self.xyz_df = None
        self.distance_matrix = None

    @property
    def natoms(self):
        """Dynamically calculate the number of atoms based on the length of self.atoms."""
        return len(self.atoms)

    @property
    def coords(self):
        """Dynamically retrieve coordinates from atoms."""
        return np.array([atom.coord for atom in self.atoms])

    def __repr__(self):
        return f"Mol(natoms={self.natoms})"
    
    def readfile(self, format, inputgeom):
        """
        Read geometry data from a file.
        
        Parameters:
        - inputgeom: Path to the geometry file.
        - file_format: Format of the file (e.g., 'xyz', 'mol2').
        """
        if not os.path.exists(inputgeom):
            raise FileNotFoundError(f"No such file: '{inputgeom}'")
        with open(inputgeom, 'r') as file:
            content = file.read().strip()
        self._parse_read(content, format)
        return self
    
    def readstr(self, inputgeom, file_format):
        """
        Read geometry data from a string.
        
        Parameters:
        - inputgeom: The geometry data as a string.
        - file_format: Format of the data (e.g., 'xyz', 'mol2').
        """
        content = inputgeom.strip()
        self._parse_read(content, file_format)
        return self
    
    def _parse_read(self, content, format):
        """
        Parse the geometry content based on the specified file format.
        
        Parameters:
        - content: The geometry content (string).
        - file_format: Format of the data (e.g., 'xyz', 'mol2').
        """
        if format == 'xyz':
            self.readxyz(content)
        elif format == 'mol2':
            self.readmol2(content)
        else:
            raise ValueError(f"Unsupported format: '{format}'")
    
    def readxyz(self, content):
        """
        Parse the XYZ content to create Atom objects.

        Parameters
        ----------
        content : str
            The XYZ format content as a string.
        """
        lines = content.splitlines()
        try:
            num_atoms = int(lines[0].strip())
            self.atoms = []  # Initialize or clear the atoms list

            for idx, line in enumerate(lines[2:num_atoms + 2], start=0):  # Start at index 0
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"Invalid line in XYZ data: {line}")
                element = parts[0]
                x, y, z = map(float, parts[1:])
                self.atoms.append(Atom(idx, element, [x, y, z]))  # Create Atom instances directly in self.atoms
        except (ValueError, IndexError) as e:
            raise ValueError("Invalid XYZ format") from e
                
    def readmol2(self, content):
        atoms = []
        bonds = []
        lines = content.splitlines()
        section_atom = None

        # Find the number of atoms and the start of the ATOM section
        num_atoms = None
        for idx, line in enumerate(lines):
            if line.startswith('@<TRIPOS>MOLECULE'):
                try:
                    num_atoms = int(lines[idx + 2].split()[0])  # Get the number of atoms
                except (IndexError, ValueError) as e:
                    raise ValueError(f"Could not read number of atoms from the file: {e}")
            if line.startswith('@<TRIPOS>ATOM'):
                section_atom = idx

        if num_atoms is None:
            raise ValueError("Number of atoms could not be determined from the file")

        # Initialize arrays for atoms and coordinates
        self.atoms = np.empty(num_atoms, dtype=object)
        # self.coords = np.empty((num_atoms, 3), dtype=float)

        # Read atom section
        if section_atom is not None:
            coords = lines[section_atom + 1 : section_atom + 1 + num_atoms]
            for line in coords:
                coord_list = line.split()
                atom_idx = int(coord_list[0]) - 1  # Convert to 0-based index
                element = remove_digits(coord_list[1])
                x, y, z = map(float, coord_list[2:5])

                # Create Atom object and store it
                self.atoms[atom_idx] = Atom(atom_idx, element, [x, y, z])
        
        else:
            raise ValueError("ATOM section not found in the file")
    
    def writefile(self, file_format, outputgeom):
        """
        Write geometry data to a file.

        Parameters:
        - outputgeom: Path to the output geometry file.
        - file_format: Format of the file (e.g., 'xyz', 'mol2').
        """
        with open(outputgeom, 'w') as file:
            content = self._generate_content(file_format)
            file.write(content)
        
    def writestr(self, file_format):
        """
        Generate geometry data as a string in the specified format.

        Parameters:
        - file_format: Format of the data (e.g., 'xyz', 'mol2').

        Returns:
        - A string containing the geometry data.
        """
        content = self._parse_write(file_format)
        return content

    def _parse_write(self, file_format):
        """
        Generate the geometry content based on the specified file format.

        Parameters:
        - file_format: Format of the data (e.g., 'xyz', 'mol2').

        Returns:
        - The content string in the specified format.
        """
        if file_format == 'xyz':
            return self.writexyz()
        elif file_format == 'mol2':
            return self.writemol2()
        else:
            raise ValueError(f"Unsupported format: '{file_format}'")

    def writexyz(self):
        """
        Generate XYZ format content for the geometry.

        Returns:
        - A string containing the XYZ format of the geometry.
        """
        num_atoms = len(self.atoms)
        lines = [f"{num_atoms}", ""]  # XYZ format starts with number of atoms and a comment line
        for atom in self.atoms:
            lines.append(f"{atom.sym}\t{atom.coord[0]:.6f}\t{atom.coord[1]:.6f}\t{atom.coord[2]:.6f}")
        return "\n".join(lines)

    def writemol2(self):
        """
        Generate MOL2 format content for the geometry.

        Returns:
        - A string containing the MOL2 format of the geometry.
        """
        lines = ["@<TRIPOS>MOLECULE", "GeneratedMol", f"{len(self.atoms)} 0 0 0 0", ""]
        lines.append("@<TRIPOS>ATOM")
        for atom in self.atoms:
            lines.append(f"{atom.idx} {atom.sym} {atom.coord[0]:.6f} {atom.coord[1]:.6f} {atom.coord[2]:.6f} {atom.sym} 1 RES1 0.0000")
        lines.append("@<TRIPOS>BOND")
        return "\n".join(lines)
   
    def get_coord_byidx(self, idx):
        coord = self.atoms[idx].coord
        return np.array(coord)
    
    def atom_byidx(self, idx):
        return self.atoms[idx]
    
    def get_listofatomprop(self, prop):
        """
        Collects the specified property from each Atom object in self.atoms.

        Parameters
        ----------
        prop : str
            The property name to retrieve from each Atom object.

        Returns
        -------
        np.ndarray
            An array containing the values of the specified property from each Atom.

        Raises
        ------
        ValueError
            If an Atom does not have the specified property.
        """
        prop_list = []  # Collect the specified properties
        for atom in self.atoms:
            try:
                # Access the property or attribute directly
                prop_value = getattr(atom, prop)
                prop_list.append(prop_value)
            except AttributeError:
                raise ValueError(f"Atom object has no property '{prop}'")
        
        return np.array(prop_list)  # Return the list as a NumPy array
    
    def _build_xyz_df(self):
        """
        Build a DataFrame that contains atom indices and their coordinates.
        """
        data = {
            'atom_idx': [atom.idx for atom in self.atoms],
            'element': [atom.sym for atom in self.atoms],
            'x': [atom.coord[0] for atom in self.atoms],
            'y': [atom.coord[1] for atom in self.atoms],
            'z': [atom.coord[2] for atom in self.atoms],
        }
        self.xyz_df = pd.DataFrame(data)

    def get_distance_matrix(self):
        """
        Calculate the distance matrix using broadcasting and vectorized operations.
        
        Returns:
        - A pandas DataFrame containing the distance matrix.
        """
        if self.xyz_df is None:
            self._build_xyz_df()  # Build the DataFrame if it doesn't exist
        atom_coordinates = self.xyz_df[['x', 'y', 'z']].to_numpy()
        diff = atom_coordinates[:, np.newaxis, :] - atom_coordinates[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        self.distance_matrix = pd.DataFrame(distance_matrix)
        return self.distance_matrix
    
    def get_bond_existed(self, cutoff=0.1):
        self.get_distance_matrix()
        num_atoms = self.natoms
        atoms = self.atoms
        cov_radii = self.get_listofatomprop('cov_radii')
        bond_existed = list()
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                bond_length = self.distance_matrix.iloc[i, j]
                cov_radii_i = self.atoms[i].cov_radii  # Calculate covalent radius when needed
                cov_radii_j = self.atoms[j].cov_radii
                covalent_sum = cov_radii_i + cov_radii_j + cutoff
                
                if bond_length < covalent_sum:
                    bond_existed.append([atoms[i], atoms[j], bond_length])
        return bond_existed            
    
    def calc_distance(self, atomA_idx, atomB_idx):
        try:
            atomA_coord = self.get_coord_byidx(atomA_idx)
            atomB_coord = self.get_coord_byidx(atomB_idx)
        except KeyError as e:
            raise ValueError(f"Invalid atom index: {e}")
        dist = np.linalg.norm(atomA_coord - atomB_coord)
        return dist
    
    def calc_angle(self, atomA_idx, atomB_idx, atomC_idx):
        try:
            atomA_coord = self.get_coord_byidx(atomA_idx)
            atomB_coord = self.get_coord_byidx(atomB_idx)
            atomC_coord = self.get_coord_byidx(atomC_idx)
        except KeyError as e:
            raise ValueError(f"Invalid atom index: {e}")
        vector_BA = atomA_coord - atomB_coord
        vector_BC = atomC_coord - atomB_coord
        norm_BA = np.linalg.norm(vector_BA)
        norm_BC = np.linalg.norm(vector_BC)
        if norm_BA == 0 or norm_BC == 0:
            raise ValueError("One of the vectors has zero length, which makes the angle calculation undefined.")
        uBA = vector_BA / norm_BA
        uBC = vector_BC / norm_BC
        dot_product = np.dot(uBA, uBC)
        dot_product_clipped = np.clip(dot_product, -1.0, 1.0)
        angle_radians = np.arccos(dot_product_clipped)
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees
    
    def calc_dihedral(self, atomA_idx, atomB_idx, atomC_idx, atomD_idx):
        """
        Calculate the dihedral angle between four points A, B, C, and D in 3D space.
        Returns the angle in degrees.
        """
        try:
            atomA_coord = self.get_coord_byidx(atomA_idx)
            atomB_coord = self.get_coord_byidx(atomB_idx)
            atomC_coord = self.get_coord_byidx(atomC_idx)
            atomD_coord = self.get_coord_byidx(atomD_idx)
        except KeyError as e:
            raise ValueError(f"Invalid atom index: {e}")
        vector_BA = atomA_coord - atomB_coord
        vector_BC = atomC_coord - atomB_coord
        vector_CD = atomD_coord - atomC_coord
        n1 = np.cross(vector_BA, vector_BC)
        n2 = np.cross(vector_BC, vector_CD)
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        dot_product = np.dot(n1, n2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_radians = np.arccos(dot_product)
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees
    
    def atomidx_from_element(self, element):
        list_indices = []
        for atom in self.atoms:
            if atom.sym == element:
                list_indices.append(atom.idx)
        # list_indices = [3, 4, 14, 15, 16, 19, 21, 24, 28, 32] of the element
        return list_indices
    
    def atomnearby_from_idx(self, atomA_idx, cutoff=0.1):
        bond_existed = self.get_bond_existed(cutoff=cutoff)
        nearby_atoms = set()
        for bond in bond_existed:
            if atomA_idx in [bond[0].idx, bond[1].idx]:  # Check if atomA_idx is part of the bond
                atom1_idx = int(bond[0].idx)
                atom2_idx = int(bond[1].idx)
                if atom1_idx != atomA_idx:
                    nearby_atoms.add(self.atoms[atom1_idx])  # Add Atom object based on its index
                if atom2_idx != atomA_idx:
                    nearby_atoms.add(self.atoms[atom2_idx])  # Add Atom object based on its index
        return nearby_atoms
    
    def get_unique_elements(self):
        unique_elements = {atom.sym for atom in self.atoms}
        return unique_elements
    
    def get_transition_metal(self):
        unique_elements = self.get_unique_elements()
        metals = unique_elements.intersection(TRANSITION_METALS)
        if len(metals) == 0:
            return None
        return metals  # Set of metals, e.g., {'Ni', 'Pd'}
    
    def isbond(self, atomA_idx, atomB_idx, cutoff=0.1):
        cov_radii_A = self.atoms[atomA_idx].cov_radii
        cov_radii_B = self.atoms[atomB_idx].cov_radii
        distance = self.calc_distance(atomA_idx, atomB_idx)
        isbond = distance < (cov_radii_A + cov_radii_B + cutoff)
        return isbond

    def sum_cov_radii(self, elementA, elementB, cutoff=0.0):
        cov_radii_A = COVALENT_RADII[elementA]
        cov_radii_B = COVALENT_RADII[elementB]
        sum_radii = cov_radii_A + cov_radii_B + cutoff
        return sum_radii

    def find_ligand_mononuclear(self, getonlyidx=False):
        metal_element = list(self.get_transition_metal())[0]
        metal_idx = self.atomidx_from_element(metal_element)[0]
        bondtometal = self.atomnearby_from_idx(metal_idx)
        if getonlyidx:
            bondtometal = {atom.idx for atom in bondtometal}
        return bondtometal

    def find_nearest_ofelement_toatom(self, center_atom_index, target_element, cutoff=0.1):
        bond_to_center_atom = self.atomnearby_from_idx(center_atom_index, cutoff)
        atoms_with_target_element = [atom for atom in bond_to_center_atom if atom.sym == target_element]
        nearest_atom = min(atoms_with_target_element, key=lambda atom: self.calc_distance(center_atom_index, atom.idx))
        return nearest_atom

    def is_planar(self, list_of_atoms, tolerance=5):
        if len(list_of_atoms) < 4:
            raise ValueError("At least 4 atom indices are required to check for planarity.")
        atomA_idx, atomB_idx, atomC_idx = list_of_atoms[0], list_of_atoms[1], list_of_atoms[2]
        for i in range(3, len(list_of_atoms)):
            atomD_idx = list_of_atoms[i]
            dihedral = self.calc_dihedral(atomA_idx, atomB_idx, atomC_idx, atomD_idx)
            if abs(dihedral) > tolerance and abs(dihedral - 180) > tolerance:
                return False
        return True

    def is_sqp(self, tolerance=5):
        if len(self.find_ligand_mononuclear()) != 4:
            raise ValueError("Square planar geometry requires exactly 4 ligands.")
        ligatoms = list(self.find_ligand_mononuclear())
        atomA_idx, atomB_idx, atomC_idx, atomD_idx = ligatoms[0].idx, ligatoms[1].idx, ligatoms[2].idx, ligatoms[3].idx
        angle = self.calc_dihedral(atomA_idx, atomB_idx, atomC_idx, atomD_idx)
        return abs(angle) < tolerance or abs(angle - 180) < tolerance

    def find_normv(self, list_of_atoms):
        if len(list_of_atoms) != 3:
            raise ValueError('You need exactly 3 atoms to find the normal vector of a plane.')
        atomA_idx, atomB_idx, atomC_idx = list_of_atoms
        atomA_coord = self.get_coord_byidx(atomA_idx)
        atomB_coord = self.get_coord_byidx(atomB_idx)
        atomC_coord = self.get_coord_byidx(atomC_idx)
        vector_BA = atomA_coord - atomB_coord
        vector_BC = atomC_coord - atomB_coord
        normv = np.cross(vector_BA, vector_BC)
        normv = normv / np.linalg.norm(normv)
        if normv[0] < 0:
            normv = -normv
        return normv

    def angle_twovec(self, vector_BA, vector_BC):
        norm_BA = np.linalg.norm(vector_BA)
        norm_BC = np.linalg.norm(vector_BC)
        if norm_BA == 0 or norm_BC == 0:
            raise ValueError("One of the vectors has zero length, which makes the angle calculation undefined.")
        uBA = vector_BA / norm_BA
        uBC = vector_BC / norm_BC
        dot_product = np.dot(uBA, uBC)
        dot_product_clipped = np.clip(dot_product, -1.0, 1.0)
        angle_radians = np.arccos(dot_product_clipped)
        return np.degrees(angle_radians)

    def reindex_atoms(self):
        """Reindex atoms based on their position in self.atoms."""
        for idx, atom in enumerate(self.atoms):
            atom.idx = idx
    
    def addAtom(self, atom, reindex=True):
        """
        Adds an Atom to the molecule and optionally reindexes all atoms.

        Parameters
        ----------
        atom : Atom
            The Atom instance to add to the molecule.
        reindex : bool, optional
            If True, reindex all atoms after adding. Defaults to True.
        """
        self.atoms.append(atom)
        if reindex:
            self.reindex_atoms()
        
        
    def deleteAtom(self, idx):
        """
        Deletes an Atom from the molecule by index and reindexes the remaining atoms.

        Parameters
        ----------
        idx : int
            The index of the Atom to delete.
        """
        atom_to_delete = None
        for atom in self.atoms:
            if atom.idx == idx:
                atom_to_delete = atom
                break
        if atom_to_delete is None:
            raise ValueError(f"No Atom with index {idx} found.")
        self.atoms = [atom for atom in self.atoms if atom.idx != idx]
        self.reindex_atoms()  # Update indices
    
    def deleteAtoms(self, indices):
        """
        Deletes a list of Atoms from the molecule based on their indices.

        Parameters
        ----------
        indices : list of int
            The list of indices of Atoms to delete.

        Raises
        ------
        ValueError
            If any Atom with a specified index is not found.
        """
        indices_set = set(indices)
        self.atoms = [atom for atom in self.atoms if atom.idx not in indices_set]
        self.reindex_atoms()
    
    def deleteAtoms_bysym(self, symbols):
        """
        Deletes all atoms with specified element symbols from the molecule.

        Parameters
        ----------
        symbols : list of str
            List of element symbols (e.g., ['Fe', 'X']) to delete from the molecule.

        Raises
        ------
        ValueError
            If no atoms with the specified symbols are found.
        """
        if not isinstance(symbols, list):
            raise ValueError("The symbols parameter must be a list of element symbols (strings).")
        symbols_set = set(symbols)
        atoms_to_keep = [atom for atom in self.atoms if atom.sym not in symbols_set]
        if len(atoms_to_keep) == len(self.atoms):
            raise ValueError(f"No atoms with symbols {symbols} found.")
        self.atoms = atoms_to_keep
        self._coords = np.array([atom.coord for atom in self.atoms]) if self.atoms else None
        self.reindex_atoms()

    def getAtom_fromidx(self, idx):
        matching_atoms = [atom for atom in self.atoms if atom.idx == idx]
        if len(matching_atoms) > 1:
            raise ValueError(f"More than one atom found with index {idx}.")
        elif len(matching_atoms) == 0:
            raise ValueError(f"No atom found with index {idx}.")
        return matching_atoms[0]
    
    def getlistofsym(self):
        list_sym = []
        for atom in self.atoms:
            sym = atom.sym
            list_sym.append(sym)
        return list_sym
    
    def getcoord_fromatomlist(self, list_Atoms):
        """
        Get the coordinates of atoms in the provided list.

        Parameters
        ----------
        list_Atoms : list
            List of Atom objects from which to extract coordinates.

        Returns
        -------
        np.ndarray
            A 2D numpy array with each row representing the coordinates of an atom.
        
        Raises
        ------
        ValueError
            If the input is not a list.
        """
        if not isinstance(list_Atoms, list):
            raise ValueError("The input must be a list of Atom objects.")
        
        coords = np.vstack([atom.coord for atom in list_Atoms])
        return coords
    
    def setnewcoord(self, idx, newcoord):
        self.atoms
    
class MolLigand(Mol):
    def __init__(self):
        """
        Initialize the MolLigand object with optional attributes for name, path, and SMILES.
        
        Parameters
        ----------
        name : str, optional
            Name of the molecule.
        path : str, optional
            File path for the structure, if specified.
        smi : str, optional
            SMILES string for the molecule, only when needed.
        """
        super().__init__()
        self.catoms = []  # List of critical atoms, input from the user
        self.name = None
        self.smi = None  # SMILES string if specified
    
    def __repr__(self):
        return f"MolLigand(natoms={self.natoms}, catoms={self.catoms})"
    
    def set_catoms(self, catoms):
        """
        Set the critical atoms (catoms) for the molecule.
        
        Parameters
        ----------
        catoms : list
            List of atom indices or identifiers provided by the user.
        """
        if not isinstance(catoms, list):
            raise ValueError("catoms must be a list.")
        self.catoms = catoms
    
    def set_name(self, name):
        """
        Set the name of the molecule.
        
        Parameters
        ----------
        name : str
            The name to assign to the molecule.
        """
        if not isinstance(name, str):
            raise ValueError("The name must be a string.")
        self.name = name
        
    def getAtom_catoms(self):
        """
        Get the Atom instances corresponding to the indices in catoms.

        Returns
        -------
        List[Atom]
            List of Atom instances for the critical atoms (catoms).

        Raises
        ------
        ValueError
            If catoms is not set or if any index in catoms does not correspond to an atom in atoms.
        """
        if self.catoms is None or len(self.catoms) == 0:
            raise ValueError("catoms must be set before calling getcatoms_Atom.")

        catom_atoms = [atom for atom in self.atoms if atom.idx in self.catoms]

        # Ensure all catoms indices have corresponding Atom objects
        if len(catom_atoms) != len(self.catoms):
            raise ValueError("Not all indices in catoms correspond to atoms in the molecule.")

        return set(catom_atoms)
            
    def getAtom_adjcatom(self):
        """
        Get a dictionary of critical atoms (catoms) and their adjacent atoms.

        Returns
        -------
        dict
            A dictionary where each key is the index of a critical atom (catom) and the value
            is a list of adjacent atoms.

        Raises
        ------
        ValueError
            If catoms is not set or if any index in catoms does not correspond to an atom in atoms.
        """
        if not self.catoms:
            raise ValueError("catoms must be set before calling getAtom_adjcatom.")
        
        adjcatoms = {}
        for idx_catom in self.catoms:
            atomnearby = list(self.atomnearby_from_idx(idx_catom))
            adjcatoms[idx_catom] = atomnearby
        return adjcatoms
    
        
        
        
