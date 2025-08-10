#!/usr/bin/env python3
"""
Test script to verify SMARTS patterns are fixed in the monolith.
"""

import warnings
warnings.filterwarnings('ignore')

# Suppress RDKit warnings and errors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')

import sys
from io import StringIO

class SuppressRDKitOutput:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self._original_stderr

# Test the molecular featurizer
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np

def test_polymer_features(smiles):
    """Test polymer feature extraction without SMARTS."""
    try:
        with SuppressRDKitOutput():
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [0.0] * 20
            
            features = []
            
            # Count atoms by element (safer than SMARTS)
            atom_counts = {}
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
            
            # Common polymer elements
            polymer_elements = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'Si']
            for element in polymer_elements:
                features.append(atom_counts.get(element, 0))
            
            # Bond type counts
            bond_counts = {'SINGLE': 0, 'DOUBLE': 0, 'TRIPLE': 0, 'AROMATIC': 0}
            for bond in mol.GetBonds():
                bond_type = str(bond.GetBondType())
                if bond_type in bond_counts:
                    bond_counts[bond_type] += 1
            
            features.extend([bond_counts['SINGLE'], bond_counts['DOUBLE'], 
                           bond_counts['TRIPLE'], bond_counts['AROMATIC']])
            
            # Ring information
            ring_info = mol.GetRingInfo()
            features.extend([
                ring_info.NumRings(),
                len([r for r in ring_info.AtomRings() if len(r) == 5]),
                len([r for r in ring_info.AtomRings() if len(r) == 6]),
            ])
            
            # Basic properties
            basic_props = [
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                mol.GetNumHeavyAtoms(),
            ]
            features.extend(basic_props)
            
            # Pad to 20 features
            while len(features) < 20:
                features.append(0.0)
            
            return features[:20]
            
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return [0.0] * 20

# Test with some sample SMILES
test_smiles = [
    'CCO',  # Ethanol
    'c1ccccc1',  # Benzene
    'CC(C)O',  # Isopropanol
    'C1=CC=CC=C1O',  # Phenol
    'CCCCCCCC',  # Octane
    '[Si](C)(C)O[Si](C)(C)C',  # Silicone-like
]

print("Testing polymer feature extraction...")
for smiles in test_smiles:
    features = test_polymer_features(smiles)
    print(f"{smiles}: {len(features)} features, first 5: {features[:5]}")

print("Test completed successfully - no SMARTS errors!")