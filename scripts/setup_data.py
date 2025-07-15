#!/usr/bin/env python3
"""Script to set up sample data for the polymer prediction project."""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_data(n_samples: int = 1000, output_path: str = "data/sample_data.csv"):
    """Generate sample polymer data for testing and development.
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the generated data
    """
    # Sample SMILES strings for different polymer types
    sample_smiles = [
        "CC(C)c1ccccc1",  # Cumene
        "c1ccccc1",       # Benzene
        "CCO",            # Ethanol
        "C=CC=C",         # Butadiene
        "c1cnccc1",       # Pyridine
        "O=C(O)c1ccccc1", # Benzoic acid
        "CN(C)C=O",       # DMF
        "C1CCCCC1",       # Cyclohexane
        "CC(C)(C)c1ccccc1", # tert-Butylbenzene
        "OC(=O)c1ccc(C(=O)O)cc1", # Terephthalic acid
        "CC(C)(C)OC(=O)C=C", # tert-Butyl acrylate
        "C=Cc1ccccc1",    # Styrene
        "CC(=C)C(=O)OC",  # Methyl methacrylate
        "C=CC(=O)O",      # Acrylic acid
        "C=CCl",          # Vinyl chloride
        "C=CC#N",         # Acrylonitrile
        "CC(C)=CC=C(C)C", # Myrcene
        "c1ccc2ccccc2c1", # Naphthalene
        "CC1=CC=C(C=C1)C", # p-Xylene
        "CC1=CC=CC=C1C",  # o-Xylene
    ]
    
    # Generate random data
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        # Randomly select a SMILES string
        smiles = np.random.choice(sample_smiles)
        
        # Generate a synthetic target property
        # This could represent glass transition temperature, melting point, etc.
        base_value = np.random.normal(100, 50)
        
        # Add some structure-property relationship
        if "c1ccccc1" in smiles:  # Aromatic compounds tend to have higher values
            base_value += 30
        if "C=C" in smiles:  # Unsaturated compounds
            base_value += 10
        if "O" in smiles:  # Oxygen-containing compounds
            base_value -= 20
        
        # Add noise
        target_value = base_value + np.random.normal(0, 10)
        
        data.append({
            "smiles": smiles,
            "target_property": round(target_value, 2),
            "sample_id": f"sample_{i:04d}",
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} samples and saved to {output_path}")
    
    # Print some statistics
    print(f"\nData Statistics:")
    print(f"Target property range: {df['target_property'].min():.2f} to {df['target_property'].max():.2f}")
    print(f"Target property mean: {df['target_property'].mean():.2f}")
    print(f"Target property std: {df['target_property'].std():.2f}")
    print(f"Unique SMILES: {df['smiles'].nunique()}")


def create_train_test_split(
    input_path: str = "data/sample_data.csv",
    train_path: str = "data/train.csv",
    test_path: str = "data/test.csv",
    test_size: float = 0.2,
):
    """Split data into train and test sets.
    
    Args:
        input_path: Path to input data
        train_path: Path to save training data
        test_path: Path to save test data
        test_size: Fraction of data to use for testing
    """
    # Load data
    df = pd.read_csv(input_path)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    n_test = int(len(df) * test_size)
    test_df = df[:n_test]
    train_df = df[n_test:]
    
    # Save splits
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Created train set with {len(train_df)} samples: {train_path}")
    print(f"Created test set with {len(test_df)} samples: {test_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up sample data for polymer prediction")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output_path", type=str, default="data/sample_data.csv", help="Output path for generated data")
    parser.add_argument("--split", action="store_true", help="Create train/test split")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size fraction")
    
    args = parser.parse_args()
    
    # Generate sample data
    generate_sample_data(args.n_samples, args.output_path)
    
    # Create train/test split if requested
    if args.split:
        create_train_test_split(
            args.output_path,
            test_size=args.test_size
        )