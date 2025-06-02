"""
Data utility functions for preprocessing and data handling.
Functional programming approach for modularity and reusability.
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


def detect_data_type(data_path: str) -> str:
    """Auto-detect data type based on file structure."""
    data_path = Path(data_path)
    
    # Check for image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    has_images = any(
        f.suffix.lower() in image_extensions 
        for f in data_path.rglob('*') if f.is_file()
    )
    
    # Check for text/CSV files
    text_extensions = {'.csv', '.txt', '.json'}
    has_text = any(
        f.suffix.lower() in text_extensions 
        for f in data_path.rglob('*') if f.is_file()
    )
    
    if has_images:
        return 'image'
    elif has_text:
        # Try to determine if it's text classification or tabular
        csv_files = list(data_path.glob('*.csv'))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            text_columns = [col for col in df.columns 
                          if df[col].dtype == 'object' and 
                          df[col].str.len().mean() > 50]  # Likely text
            if text_columns:
                return 'text'
            else:
                return 'tabular'
    
    return 'unknown'


def load_and_analyze_data(data_path: str) -> Dict[str, Any]:
    """Load and analyze dataset to provide insights."""
    data_type = detect_data_type(data_path)
    analysis = {'data_type': data_type}
    
    if data_type == 'image':
        analysis.update(_analyze_image_data(data_path))
    elif data_type in ['text', 'tabular']:
        analysis.update(_analyze_tabular_data(data_path))
    
    return analysis


def _analyze_image_data(data_path: str) -> Dict[str, Any]:
    """Analyze image dataset structure."""
    data_path = Path(data_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Count images by class (if organized in folders)
    class_counts = {}
    total_images = 0
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            count = len([f for f in class_dir.iterdir() 
                        if f.suffix.lower() in image_extensions])
            if count > 0:
                class_counts[class_dir.name] = count
                total_images += count
    
    # If no class structure, count all images
    if not class_counts:
        all_images = [f for f in data_path.rglob('*') 
                     if f.suffix.lower() in image_extensions]
        total_images = len(all_images)
    
    return {
        'num_classes': len(class_counts) if class_counts else 1,
        'class_counts': class_counts,
        'total_images': total_images,
        'balanced': _check_balance(class_counts) if class_counts else True
    }


def _analyze_tabular_data(data_path: str) -> Dict[str, Any]:
    """Analyze tabular/text dataset."""
    data_path = Path(data_path)
    csv_files = list(data_path.glob('*.csv'))
    
    if not csv_files:
        return {'error': 'No CSV files found'}
    
    df = pd.read_csv(csv_files[0])
    
    # Basic stats
    analysis = {
        'num_samples': len(df),
        'num_features': len(df.columns),
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict()
    }
    
    # Check for target column
    possible_targets = ['label', 'target', 'class', 'y']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    if target_col:
        analysis['target_column'] = target_col
        analysis['num_classes'] = df[target_col].nunique()
        analysis['class_distribution'] = df[target_col].value_counts().to_dict()
        analysis['balanced'] = _check_balance(analysis['class_distribution'])
    
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
                       num_workers: int = 4,
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
                               save_path: Optional[str] = None) -> None:
    """Visualize data distribution and characteristics."""
    data_type = data_analysis['data_type']
    
    if data_type == 'image' and 'class_counts' in data_analysis:
        _plot_class_distribution(data_analysis['class_counts'], 
                                'Image Classes', save_path)
    
    elif data_type in ['text', 'tabular'] and 'class_distribution' in data_analysis:
        _plot_class_distribution(data_analysis['class_distribution'], 
                                'Class Distribution', save_path)


def _plot_class_distribution(class_counts: Dict[str, int], 
                            title: str, 
                            save_path: Optional[str] = None) -> None:
    """Plot class distribution."""
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
    plt.show()


def create_submission_file(predictions: np.ndarray, 
                          submission_format: str = 'simple',
                          output_file: str = 'submission.csv') -> None:
    """Create submission file in common competition formats."""
    
    if submission_format == 'simple':
        # Simple format: just predictions
        np.savetxt(output_file, predictions, fmt='%d')
        
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
            'predicted_class': predictions
        })
        df.to_csv(output_file, index=False)
    
    print(f'Submission file saved as {output_file}')


def get_recommended_config(data_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Get recommended configuration based on data analysis."""
    data_type = data_analysis['data_type']
    config = {'data_type': data_type}
    
    if data_type == 'image':
        num_classes = data_analysis.get('num_classes', 2)
        total_images = data_analysis.get('total_images', 1000)
        
        config.update({
            'model_type': 'image_classifier',
            'backbone': 'resnet18' if total_images < 10000 else 'resnet50',
            'batch_size': 32 if total_images < 5000 else 64,
            'learning_rate': 1e-3,
            'num_epochs': 50 if total_images < 5000 else 100,
            'early_stopping': {'patience': 10}
        })
        
    elif data_type == 'text':
        num_samples = data_analysis.get('num_samples', 1000)
        
        config.update({
            'model_type': 'text_classifier',
            'model_name': 'bert-base-uncased',
            'max_length': 256,
            'batch_size': 16 if num_samples < 10000 else 32,
            'learning_rate': 2e-5,
            'num_epochs': 10,
            'early_stopping': {'patience': 3}
        })
        
    elif data_type == 'tabular':
        num_features = data_analysis.get('num_features', 10)
        num_samples = data_analysis.get('num_samples', 1000)
        
        config.update({
            'model_type': 'mlp',
            'hidden_dims': [256, 128] if num_features < 50 else [512, 256, 128],
            'batch_size': 64,
            'learning_rate': 1e-3,
            'num_epochs': 100,
            'early_stopping': {'patience': 15}
        })
    
    return config 