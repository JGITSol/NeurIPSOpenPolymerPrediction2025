"""Featurization utilities for converting SMILES to graph representations."""

import torch
from rdkit import Chem
from torch_geometric.data import Data


def get_atom_features(atom):
    """Extract features for a single atom.
    
    Args:
        atom (rdkit.Chem.Atom): RDKit atom object
        
    Returns:
        list: List of atom features
    """
    # These features are standard choices for molecular GNNs
    features = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
        atom.GetIsAromatic(),
        atom.GetChiralTag().real,
    ]
    
    # One-hot encode atomic number (example for C, O, N)
    # A more robust implementation would handle all possible elements
    atomic_num_one_hot = [0] * 3
    if atom.GetAtomicNum() == 6:
        atomic_num_one_hot[0] = 1
    elif atom.GetAtomicNum() == 8:
        atomic_num_one_hot[1] = 1
    elif atom.GetAtomicNum() == 7:
        atomic_num_one_hot[2] = 1
    
    return features + atomic_num_one_hot


def smiles_to_graph(smiles_string):
    """Convert a SMILES string to a PyG Data object.
    
    Args:
        smiles_string (str): SMILES representation of a molecule
        
    Returns:
        torch_geometric.data.Data: Graph representation of the molecule,
            or None if parsing fails
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)  # Add explicit hydrogens

    # Get atom features
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # Get bond features and connectivity
    if mol.GetNumBonds() > 0:
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append((i, j))
            edge_indices.append((j, i))  # Graph must be undirected

            # Example bond features
            bond_type = bond.GetBondTypeAsDouble()
            is_in_ring = bond.IsInRing()
            edge_attrs.append([bond_type, is_in_ring])
            edge_attrs.append([bond_type, is_in_ring])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Handle molecules with no bonds (e.g., single atoms)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float)

    # Create the PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Store number of features for model instantiation later
    data.num_atom_features = x.size(1)

    return data
