"""I/O utility functions for polymer prediction."""

import os
import json
import pickle
import pandas as pd
import torch
import numpy as np


def save_model(model, path, metadata=None):
    """Save a model to disk.
    
    Args:
        model (torch.nn.Module): Model to save
        path (str): Path to save the model to
        metadata (dict, optional): Additional metadata to save with the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if metadata is not None:
        save_dict['metadata'] = metadata
        
    torch.save(save_dict, path)
    
    
def load_model(model_class, path, device=None):
    """Load a model from disk.
    
    Args:
        model_class (type): Model class to instantiate
        path (str): Path to load the model from
        device (torch.device, optional): Device to load the model to
        
    Returns:
        tuple: (model, metadata)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    checkpoint = torch.load(path, map_location=device)
    
    # Extract metadata if it exists
    metadata = checkpoint.get('metadata', None)
    
    # Instantiate the model using metadata if available
    if metadata and 'model_params' in metadata:
        model = model_class(**metadata['model_params'])
    else:
        # If no metadata, assume default parameters
        model = model_class()
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, metadata


class NumpyTorchEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and PyTorch types."""
    
    def default(self, obj):
        # Handle PyTorch tensors
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle numpy numeric types
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
            
        # Let the base class handle it
        return super().default(obj)


def save_results(results, path):
    """Save results to disk.
    
    Args:
        results (dict): Results to save
        path (str): Path to save the results to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Determine file format based on extension
    _, ext = os.path.splitext(path)
    
    if ext == '.json':
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyTorchEncoder)
    elif ext == '.pkl':
        with open(path, 'wb') as f:
            pickle.dump(results, f)
    elif ext in ['.csv', '.tsv']:
        sep = ',' if ext == '.csv' else '\t'
        pd.DataFrame(results).to_csv(path, sep=sep, index=False)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def load_data(path):
    """Load data from disk.
    
    Args:
        path (str): Path to load the data from
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    _, ext = os.path.splitext(path)
    
    if ext == '.csv':
        return pd.read_csv(path)
    elif ext == '.tsv':
        return pd.read_csv(path, sep='\t')
    elif ext == '.json':
        return pd.read_json(path)
    elif ext == '.pkl':
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
