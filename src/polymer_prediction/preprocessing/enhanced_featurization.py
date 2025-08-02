"""Enhanced molecular featurization for GPU-accelerated solution."""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, Descriptors, Crippen, Lipinski
from torch_geometric.data import Data
from typing import List, Dict, Any


# Enhanced atom and bond property mappings
ATOM_PROPS = {
    'atomic_num': list(range(1, 119)),
    'chirality': [int(x) for x in Chem.rdchem.ChiralType.values],
    'degree': list(range(0, 9)),
    'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
    'hybridization': [int(x) for x in Chem.rdchem.HybridizationType.values],
    'aromatic': [0, 1],
    'in_ring': [0, 1],
    'num_hs': list(range(0, 5)),
}

BOND_PROPS = {
    'bond_type': [int(x) for x in Chem.rdchem.BondType.values],
    'conjugated': [0, 1],
    'stereo': [int(x) for x in Chem.rdchem.BondStereo.values],
    'in_ring': [0, 1],
}


def one_hot_encode(value: Any, choices: List[Any]) -> List[int]:
    """One-hot encode a value given a list of choices."""
    encoding = [0] * (len(choices) + 1)  # +1 for unknown
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1  # unknown category
    return encoding


def get_enhanced_atom_features(atom: Chem.Atom) -> List[float]:
    """Extract enhanced features for a single atom."""
    features = []
    
    # Basic properties
    features.extend(one_hot_encode(atom.GetAtomicNum(), ATOM_PROPS['atomic_num']))
    features.extend(one_hot_encode(int(atom.GetChiralTag()), ATOM_PROPS['chirality']))
    features.extend(one_hot_encode(atom.GetDegree(), ATOM_PROPS['degree']))
    features.extend(one_hot_encode(atom.GetFormalCharge(), ATOM_PROPS['formal_charge']))
    features.extend(one_hot_encode(int(atom.GetHybridization()), ATOM_PROPS['hybridization']))
    features.extend(one_hot_encode(int(atom.GetIsAromatic()), ATOM_PROPS['aromatic']))
    features.extend(one_hot_encode(int(atom.IsInRing()), ATOM_PROPS['in_ring']))
    features.extend(one_hot_encode(atom.GetTotalNumHs(), ATOM_PROPS['num_hs']))
    
    # Additional chemical properties
    features.append(atom.GetMass())
    features.append(atom.GetTotalValence())
    features.append(float(atom.IsInRingSize(3)))
    features.append(float(atom.IsInRingSize(4)))
    features.append(float(atom.IsInRingSize(5)))
    features.append(float(atom.IsInRingSize(6)))
    features.append(float(atom.IsInRingSize(7)))
    features.append(float(atom.IsInRingSize(8)))
    
    return features


def get_enhanced_bond_features(bond: Chem.Bond) -> List[float]:
    """Extract enhanced features for a single bond."""
    features = []
    
    features.extend(one_hot_encode(int(bond.GetBondType()), BOND_PROPS['bond_type']))
    features.extend(one_hot_encode(int(bond.GetIsConjugated()), BOND_PROPS['conjugated']))
    features.extend(one_hot_encode(int(bond.GetStereo()), BOND_PROPS['stereo']))
    features.extend(one_hot_encode(int(bond.IsInRing()), BOND_PROPS['in_ring']))
    
    # Additional bond properties
    features.append(float(bond.IsInRingSize(3)))
    features.append(float(bond.IsInRingSize(4)))
    features.append(float(bond.IsInRingSize(5)))
    features.append(float(bond.IsInRingSize(6)))
    features.append(float(bond.IsInRingSize(7)))
    features.append(float(bond.IsInRingSize(8)))
    
    return features


def smiles_to_enhanced_graph(smiles_string: str) -> Data:
    """Convert a SMILES string to an enhanced PyG Data object."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    
    # Add hydrogens for complete representation
    mol = Chem.AddHs(mol)
    
    # Get enhanced atom features
    atom_features = [get_enhanced_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Get enhanced bond features and connectivity
    if mol.GetNumBonds() > 0:
        edge_indices = []
        edge_attrs = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # Add both directions for undirected graph
            edge_indices.extend([(i, j), (j, i)])
            
            bond_features = get_enhanced_bond_features(bond)
            edge_attrs.extend([bond_features, bond_features])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Handle molecules with no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        # Create empty edge attributes with correct dimension
        edge_attr = torch.empty((0, 20), dtype=torch.float)  # 20 is the bond feature dimension
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_atom_features = x.size(1)
    data.num_bond_features = edge_attr.size(1) if edge_attr.size(0) > 0 else 0
    
    return data


def get_molecular_descriptors(smiles: str) -> np.ndarray:
    """Extract molecular descriptors for tabular ensemble."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(20)  # Return zeros for invalid SMILES
    
    descriptors = []
    
    # Basic descriptors
    descriptors.append(Descriptors.MolWt(mol))
    descriptors.append(Descriptors.MolLogP(mol))
    descriptors.append(Descriptors.NumHDonors(mol))
    descriptors.append(Descriptors.NumHAcceptors(mol))
    descriptors.append(Descriptors.TPSA(mol))
    descriptors.append(Descriptors.NumRotatableBonds(mol))
    descriptors.append(Descriptors.NumAromaticRings(mol))
    descriptors.append(Descriptors.NumSaturatedRings(mol))
    descriptors.append(Descriptors.NumAliphaticRings(mol))
    descriptors.append(Descriptors.RingCount(mol))
    
    # Lipinski descriptors
    descriptors.append(Lipinski.NumHDonors(mol))
    descriptors.append(Lipinski.NumHAcceptors(mol))
    descriptors.append(Lipinski.NumRotatableBonds(mol))
    
    # Crippen descriptors
    descriptors.append(Crippen.MolLogP(mol))
    descriptors.append(Crippen.MolMR(mol))
    
    # Additional descriptors
    descriptors.append(Descriptors.BertzCT(mol))
    descriptors.append(Descriptors.BalabanJ(mol))
    descriptors.append(Descriptors.HallKierAlpha(mol))
    descriptors.append(Descriptors.Kappa1(mol))
    descriptors.append(Descriptors.Kappa2(mol))
    
    return np.array(descriptors, dtype=np.float32)


def batch_featurize_smiles(smiles_list: List[str]) -> np.ndarray:
    """Batch featurize SMILES for tabular models."""
    features = []
    for smiles in smiles_list:
        features.append(get_molecular_descriptors(smiles))
    return np.array(features)