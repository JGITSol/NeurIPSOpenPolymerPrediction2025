"""Featurization utilities for converting SMILES to graph representations."""

import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data


def get_atom_features(atom):
    """Extract features for a single atom.
    
    Args:
        atom (rdkit.Chem.Atom): RDKit atom object
        
    Returns:
        list: List of atom features
    """
    # Basic atom features
    features = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetTotalValence(),
        int(atom.GetIsAromatic()),
        int(atom.GetChiralTag()),
        atom.GetFormalCharge(),
        int(atom.IsInRing()),
    ]
    
    # One-hot encode common atomic numbers in polymers
    # C, N, O, F, S, Cl, Br, I, P, Si, and others
    common_atoms = [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H, C, N, O, F, Si, P, S, Cl, Br, I
    atomic_num_one_hot = [0] * (len(common_atoms) + 1)  # +1 for "other"
    
    atomic_num = atom.GetAtomicNum()
    if atomic_num in common_atoms:
        atomic_num_one_hot[common_atoms.index(atomic_num)] = 1
    else:
        atomic_num_one_hot[-1] = 1  # "other" category
    
    # Hybridization one-hot encoding
    hybridization_types = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]
    hybridization_one_hot = [0] * (len(hybridization_types) + 1)  # +1 for "other"
    
    hybridization = atom.GetHybridization()
    if hybridization in hybridization_types:
        hybridization_one_hot[hybridization_types.index(hybridization)] = 1
    else:
        hybridization_one_hot[-1] = 1
    
    return features + atomic_num_one_hot + hybridization_one_hot


def get_bond_features(bond):
    """Extract features for a single bond.
    
    Args:
        bond (rdkit.Chem.Bond): RDKit bond object
        
    Returns:
        list: List of bond features
    """
    # Bond type one-hot encoding
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bond_type_one_hot = [0] * (len(bond_types) + 1)  # +1 for "other"
    
    bond_type = bond.GetBondType()
    if bond_type in bond_types:
        bond_type_one_hot[bond_types.index(bond_type)] = 1
    else:
        bond_type_one_hot[-1] = 1
    
    # Additional bond features
    features = [
        int(bond.IsInRing()),
        int(bond.GetIsConjugated()),
    ]
    
    return features + bond_type_one_hot


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

            # Enhanced bond features
            bond_features = get_bond_features(bond)
            edge_attrs.append(bond_features)
            edge_attrs.append(bond_features)  # Same features for both directions

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Handle molecules with no bonds (e.g., single atoms)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)  # Updated for new bond features

    # Create the PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Store number of features for model instantiation later
    data.num_atom_features = x.size(1)

    return data
