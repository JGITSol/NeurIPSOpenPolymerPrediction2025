#!/usr/bin/env python3
"""
Quick test script to validate SMILES parsing and identify problematic entries.
"""

import pandas as pd
from rdkit import Chem
import sys

def validate_smiles(smiles):
    """Validate a SMILES string using RDKit."""
    if pd.isna(smiles) or smiles == '' or smiles == 'invalid_smiles':
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def test_data_files():
    """Test SMILES validation on actual data files."""
    print("Testing SMILES validation on data files...")
    
    # Test train data
    try:
        train_df = pd.read_csv('info/train.csv')
        print(f"Loaded train data: {train_df.shape}")
        
        # Check for invalid SMILES
        train_valid = train_df['SMILES'].apply(validate_smiles)
        invalid_count = (~train_valid).sum()
        
        print(f"Train data - Valid SMILES: {train_valid.sum()}, Invalid: {invalid_count}")
        
        if invalid_count > 0:
            print("Invalid SMILES in train data:")
            invalid_smiles = train_df[~train_valid]['SMILES'].tolist()
            for i, smiles in enumerate(invalid_smiles[:5]):  # Show first 5
                print(f"  {i+1}: '{smiles}'")
        
    except Exception as e:
        print(f"Error loading train data: {e}")
    
    # Test test data
    try:
        test_df = pd.read_csv('info/test.csv')
        print(f"Loaded test data: {test_df.shape}")
        
        # Check for invalid SMILES
        test_valid = test_df['SMILES'].apply(validate_smiles)
        invalid_count = (~test_valid).sum()
        
        print(f"Test data - Valid SMILES: {test_valid.sum()}, Invalid: {invalid_count}")
        
        if invalid_count > 0:
            print("Invalid SMILES in test data:")
            invalid_smiles = test_df[~test_valid]['SMILES'].tolist()
            for i, smiles in enumerate(invalid_smiles[:5]):  # Show first 5
                print(f"  {i+1}: '{smiles}'")
        
    except Exception as e:
        print(f"Error loading test data: {e}")

def test_sample_smiles():
    """Test validation on sample SMILES strings."""
    print("\nTesting sample SMILES strings...")
    
    test_smiles = [
        'CCO',  # Valid - ethanol
        'c1ccccc1',  # Valid - benzene
        'invalid_smiles',  # Invalid
        '',  # Empty
        '*CC(*)c1ccccc1',  # Polymer SMILES with connection points
        'C1=CC=CC=C1O',  # Valid - phenol
        None,  # None value
    ]
    
    for smiles in test_smiles:
        is_valid = validate_smiles(smiles)
        print(f"'{smiles}' -> {'Valid' if is_valid else 'Invalid'}")

if __name__ == "__main__":
    test_sample_smiles()
    test_data_files()