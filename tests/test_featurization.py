"""Tests for the featurization module."""

import pytest
import torch
from rdkit import Chem

from polymer_prediction.preprocessing.featurization import get_atom_features, smiles_to_graph


def test_get_atom_features():
    """Test the get_atom_features function."""
    # Create a simple molecule
    mol = Chem.MolFromSmiles("CC")
    
    # Get features for the first atom (carbon)
    atom = mol.GetAtomWithIdx(0)
    features = get_atom_features(atom)
    
    # Check that the features have the expected length
    assert len(features) == 9
    
    # Check that the atomic number is correct (carbon = 6)
    assert features[0] == 6
    
    # Check that the one-hot encoding for carbon is correct
    assert features[6] == 1  # Carbon is encoded as [1, 0, 0]
    assert features[7] == 0
    assert features[8] == 0


def test_smiles_to_graph():
    """Test the smiles_to_graph function."""
    # Test with a valid SMILES string
    smiles = "CC"
    data = smiles_to_graph(smiles)
    
    # Check that the data object is created
    assert data is not None
    
    # Check that the data object has the expected attributes
    assert hasattr(data, 'x')
    assert hasattr(data, 'edge_index')
    assert hasattr(data, 'edge_attr')
    assert hasattr(data, 'num_atom_features')
    
    # Check that the number of nodes is correct (2 carbon atoms)
    assert data.x.shape[0] == 2
    
    # Check that the number of edges is correct (1 bond, but undirected so 2 edges)
    assert data.edge_index.shape[1] == 2
    
    # Test with an invalid SMILES string
    invalid_smiles = "invalid_smiles"
    invalid_data = smiles_to_graph(invalid_smiles)
    
    # Check that None is returned for invalid SMILES
    assert invalid_data is None
