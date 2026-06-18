import os
import sys
import pytest
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

# Append project root to path to allow importing cagqsar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cagqsar import curate_molecule, get_rdkit_descriptors

def test_curate_molecule():
    """
    Validates molecular curation and salt stripping capability.
    """
    remover = SaltRemover()
    
    # Structure contains salt (HCl)
    salt_smiles = "CN(C)C(=O)c1ccccc1.Cl"
    clean_smiles, mol = curate_molecule(salt_smiles, remover)
    
    assert mol is not None
    assert "Cl" not in clean_smiles  # Checks if HCl is removed
    assert clean_smiles == "CN(C)C(=O)c1ccccc1"

    # Structure is invalid
    bad_smiles, bad_mol = curate_molecule("invalid_smiles_string", remover)
    assert bad_mol is None

def test_get_rdkit_descriptors():
    """
    Validates RDKit descriptor calculation module.
    """
    mol = Chem.MolFromSmiles("CCO")  # Ethanol
    descriptors = get_rdkit_descriptors(mol)
    
    assert isinstance(descriptors, dict)
    assert "MolWt" in descriptors
    assert float(descriptors["MolWt"]) == pytest.approx(46.07, abs=0.1)
    assert "LogP" in descriptors or "MolLogP" in descriptors
