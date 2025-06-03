"""
Data utility functions for preprocessing and data handling.
Functional programming approach for modularity and reusability.
Focused on tabular data for MLP tasks.
"""

import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_analyze_data(data_path: str, data_type: str = 'tabular', target_column: Optional[str] = None) -> Dict[str, Any]:
    """Load and analyze tabular dataset to provide insights."""
    if data_type != 'tabular':
        raise ValueError(f"Only 'tabular' data type is supported. Got: {data_type}")
    
    return _analyze_tabular_data(data_path, target_column)


def _analyze_tabular_data(data_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Analyze tabular dataset."""
    data_path = Path(data_path)
    
    # Handle both file path and directory path
    if data_path.is_file() and data_path.suffix == '.csv':
        csv_file = data_path
    else:
        csv_files = list(data_path.glob('*.csv'))
        if not csv_files:
            return {'error': 'No CSV files found', 'data_type': 'tabular'}
        csv_file = csv_files[0]
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        return {'error': f'Error reading CSV: {e}', 'data_type': 'tabular'}
    
    # Basic stats
    analysis = {
        'data_type': 'tabular',
        'num_samples': len(df),
        'num_features': len(df.columns),
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()  # Convert to string for JSON serialization
    }
    
    # Determine target column
    if target_column and target_column in df.columns:
        actual_target_col = target_column
    else:
        # Auto-detect target column
        possible_targets = ['label', 'target', 'class', 'y']
        actual_target_col = None
        for col in possible_targets:
            if col in df.columns:
                actual_target_col = col
                break
        
        if actual_target_col is None and target_column:
            print(f"Warning: Specified target column '{target_column}' not found. Available columns: {df.columns.tolist()}")
            # Use the target_column anyway if specified, even if not found (will cause error later)
            actual_target_col = target_column
    
    if actual_target_col and actual_target_col in df.columns:
        analysis['target_column'] = actual_target_col
        target_series = df[actual_target_col]
        analysis['num_classes'] = target_series.nunique()
        
        # Determine if likely classification or regression
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 20:
            analysis['task_type'] = 'regression'
            analysis['target_range'] = [float(target_series.min()), float(target_series.max())]
            analysis['target_mean'] = float(target_series.mean())
            analysis['target_std'] = float(target_series.std())
        else:
            analysis['task_type'] = 'classification'
            analysis['class_distribution'] = target_series.value_counts().to_dict()
            analysis['balanced'] = _check_balance(analysis['class_distribution'])
    else:
        print(f"Warning: No target column found. Dataset analysis may be incomplete.")
    
    # Count numeric features for model architecture recommendations
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if actual_target_col and actual_target_col in numeric_cols:
        numeric_cols.remove(actual_target_col)
    analysis['num_numeric_features'] = len(numeric_cols)
    analysis['numeric_columns'] = numeric_cols
    
    return analysis


def _check_balance(class_counts: Dict[str, int]) -> bool:
    """Check if dataset is balanced."""
    if not class_counts:
        return True
    
    counts = list(class_counts.values())
    min_count, max_count = min(counts), max(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    return imbalance_ratio <= 2.0  # Consider balanced if ratio <= 2


def create_data_splits(dataset, 
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      random_seed: int = 42) -> Tuple:
    """Create train/val/test splits."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(random_seed)
    return random_split(dataset, [train_size, val_size, test_size], 
                       generator=generator)


def create_data_loaders(train_dataset, val_dataset, test_dataset,
                       batch_size: int = 32,
                       num_workers: int = 0,  # Changed default to 0 for simpler setup
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training."""
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def preprocess_tabular_data(df: pd.DataFrame,
                           target_column: str,
                           categorical_columns: Optional[List[str]] = None,
                           numerical_columns: Optional[List[str]] = None,
                           scaling_method: str = 'standard') -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Preprocess tabular data with encoding and scaling."""
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Auto-detect column types if not provided
    if categorical_columns is None:
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    if numerical_columns is None:
        numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
    
    preprocessors = {}
    
    # Handle categorical columns
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            preprocessors[f'{col}_encoder'] = le
    
    # Handle numerical columns
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None
    
    if scaler and numerical_columns:
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
        preprocessors['numerical_scaler'] = scaler
    
    # Encode target if categorical
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        preprocessors['target_encoder'] = target_encoder
    
    return X.values.astype(np.float32), y.astype(np.int64), preprocessors


def save_preprocessors(preprocessors: Dict, save_path: str) -> None:
    """Save preprocessing objects for later use."""
    with open(save_path, 'wb') as f:
        pickle.dump(preprocessors, f)
    print(f'Preprocessors saved to {save_path}')


def load_preprocessors(save_path: str) -> Dict:
    """Load preprocessing objects."""
    with open(save_path, 'rb') as f:
        return pickle.load(f)


def visualize_data_distribution(data_analysis: Dict[str, Any], 
                               save_path: Optional[str] = None,
                               target_column: Optional[str] = None) -> None:
    """Visualize data distribution for tabular data."""
    data_type = data_analysis.get('data_type', 'tabular')
    
    if data_type != 'tabular':
        print(f"Visualization not supported for data_type: {data_type}")
        return
    
    # Plot class distribution if it's classification
    if 'class_distribution' in data_analysis:
        _plot_class_distribution(data_analysis['class_distribution'], 
                                'Class Distribution', save_path)
    elif data_analysis.get('task_type') == 'regression' and 'target_range' in data_analysis:
        _plot_regression_target_distribution(data_analysis, save_path)
    else:
        print("No suitable data distribution to visualize.")


def _plot_class_distribution(class_counts: Dict[str, int], 
                            title: str, 
                            save_path: Optional[str] = None) -> None:
    """Plot class distribution for classification."""
    plt.figure(figsize=(10, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.bar(classes, counts)
    plt.title(title)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Class distribution plot saved to {save_path}")
    plt.show()


def _plot_regression_target_distribution(data_analysis: Dict[str, Any], 
                                        save_path: Optional[str] = None) -> None:
    """Plot target distribution for regression."""
    plt.figure(figsize=(10, 6))
    
    # For regression, we can't plot the actual distribution without the data
    # So we just show the range and basic stats
    target_range = data_analysis.get('target_range', [0, 1])
    target_mean = data_analysis.get('target_mean', 0.5)
    target_std = data_analysis.get('target_std', 0.1)
    
    plt.text(0.1, 0.8, f"Target Statistics", fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"Range: [{target_range[0]:.3f}, {target_range[1]:.3f}]", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Mean: {target_mean:.3f}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Std: {target_std:.3f}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Task Type: Regression", fontsize=12, transform=plt.gca().transAxes)
    
    plt.title("Target Variable Analysis (Regression)")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Target distribution analysis saved to {save_path}")
    plt.show()


def create_submission_file(predictions: np.ndarray, 
                          submission_format: str = 'simple',
                          output_file: str = 'submission.csv') -> None:
    """Create submission file in common competition formats."""
    
    if submission_format == 'simple':
        # Simple format: just predictions
        if predictions.ndim == 1:
            np.savetxt(output_file, predictions, fmt='%g')  # Use %g for both int and float
        else:
            np.savetxt(output_file, predictions, fmt='%g')
        
    elif submission_format == 'kaggle':
        # Kaggle format: id, prediction
        df = pd.DataFrame({
            'id': range(len(predictions)),
            'prediction': predictions
        })
        df.to_csv(output_file, index=False)
        
    elif submission_format == 'indexed':
        # Indexed format with custom column names
        df = pd.DataFrame({
            'sample_id': range(len(predictions)),
            'predicted_value': predictions  # More generic name for regression/classification
        })
        df.to_csv(output_file, index=False)
    
    print(f'Submission file saved as {output_file}')


def get_recommended_config(data_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Get recommended configuration based on data analysis for MLP models."""
    data_type = data_analysis.get('data_type', 'tabular')
    
    if data_type != 'tabular':
        raise ValueError(f"Only 'tabular' data type is supported. Got: {data_type}")
    
    num_features = data_analysis.get('num_numeric_features', data_analysis.get('num_features', 10))
    num_samples = data_analysis.get('num_samples', 1000)
    task_type = data_analysis.get('task_type', 'classification')
    
    config = {
        'data_type': 'tabular',
        'model_type': 'mlp',
        'task_type': task_type,
        'hidden_dims': [128, 64] if num_features < 50 else [256, 128, 64],
        'dropout': 0.3 if num_samples < 5000 else 0.5,
        'batch_size': 32 if num_samples < 10000 else 64,
        'learning_rate': 1e-3,
        'num_epochs': 50 if num_samples < 5000 else 100,
        'early_stopping': {'patience': 10 if num_samples < 5000 else 15},
        'normalize': True
    }
    
    # Add target column if detected
    if 'target_column' in data_analysis:
        config['target_column'] = data_analysis['target_column']
    
    return config 