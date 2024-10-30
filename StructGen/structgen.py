import glob, os
import numpy as np
from ..globalvars import MYPACKAGE_DIR
from ..AnalyzeGeom import cpx, geommath

TEMPLATE_GEOM = {'sqp' : 'sqp.xyz',
                 'tsoa': 'tsoa.xyz',
                 'tst' : 'tst.xyz'}

def create_template_Mol(geom):
    geom_template_path = TEMPLATE_GEOM[geom]
    file_path = os.path.join(MYPACKAGE_DIR, 'Data/Geometries/', geom_template_path)
    template_Mol = cpx.Mol()
    template_Mol.readfile('xyz', file_path)
    return template_Mol

def add_metal(metal_center):
    """
    Modify metal centers in template_Mol based on the specified metal_center list.

    Parameters
    ----------
    metal_center : list of str
        List of metal element symbols to replace atoms with symbol 'M'.

    Returns
    -------
    template_Mol : Mol
        The modified molecule with updated metal centers.
    """
    template_Mol = create_template_Mol()
    atoms_template_M = [atom for atom in template_Mol.atoms if atom.sym == 'M']
    if len(metal_center) < len(atoms_template_M):
        raise ValueError("Not enough elements in metal_center to match the atoms with symbol 'M'.")
    for idx, atom in enumerate(atoms_template_M):
        atom.sym = metal_center[idx % len(metal_center)]
    return template_Mol  # Return the modified template_Mol
        
# def initialize_ligand()