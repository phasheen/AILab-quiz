"""
Flexible dataset implementations for different data types.
Follows functional programming principles for modularity.
"""

import os
import csv
from typing import List, Tuple, Callable, Optional, Dict, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load CSV data into DataFrame."""
    return pd.read_csv(filepath)

class TabularDataset(Dataset):
    """Dataset for tabular/structured data."""
    
    def __init__(self, 
                 data_file: str,
                 target_column: str = 'target',
                 feature_columns: Optional[List[str]] = None,
                 normalize: bool = True):
        
        self.data = load_csv_data(data_file)
        self.target_column = target_column
        
        if feature_columns is None:
            # Exclude target_column and any non-numeric columns if not specified
            numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
            feature_columns = [col for col in numeric_cols if col != target_column]
            if not feature_columns:
                 raise ValueError("No numeric feature columns found. Please specify feature_columns or ensure your CSV has numeric features.")

        self.features_df = self.data[feature_columns]
        self.features = self.features_df.values.astype(np.float32)
        self.targets = self.data[target_column].values
        
        # Infer task type for target transformation
        target_dtype = self.data[target_column].dtype
        if pd.api.types.is_numeric_dtype(target_dtype) and self.data[target_column].nunique() > 20: # Heuristic for regression
            self.target_type = torch.float32
            # For regression, ensure targets are float32 and potentially 2D (N, 1)
            self.targets = self.targets.astype(np.float32).reshape(-1, 1)
        else: # Classification
            self.target_type = torch.long
            # Attempt to label encode if targets are not already integer
            if not pd.api.types.is_integer_dtype(self.targets):
                self.targets, _ = pd.factorize(self.targets)
            self.targets = self.targets.astype(np.int64)


        if normalize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.features = scaler.fit_transform(self.features)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=self.target_type)
        )
    
    @property
    def num_features(self) -> int:
        return self.features.shape[1]


def create_dataset(data_type: str, **kwargs) -> Dataset:
    """Factory function to create datasets based on data type."""
    if data_type == 'tabular':
        return TabularDataset(**kwargs)
    else:
        raise ValueError(f"Unsupported data type for MLP focus: {data_type}. Only 'tabular' is supported.")

def get_sample_batch(dataloader: torch.utils.data.DataLoader) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Get a single batch of data from a DataLoader."""
    try:
        sample = next(iter(dataloader))
        
        if isinstance(sample, list) and len(sample) == 2: # Common case: (inputs, labels)
            inputs, labels = sample
            return inputs, labels
        elif isinstance(sample, torch.Tensor): # Case: only inputs (unsupervised)
            # This case might be less relevant for typical supervised MLP tasks
            return sample, None
        # Removed dictionary case for text data as it's no longer supported
        else:
            print(f"Warning: Unknown sample structure in get_sample_batch: {type(sample)}")
            return None, None
            
    except StopIteration:
        print("Warning: DataLoader is empty, cannot get a sample batch.")
        return None, None
    except Exception as e:
        print(f"Error getting sample batch: {e}")
        return None, None 