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
from PIL import Image
import nltk


def load_image(path: str) -> Image.Image:
    """Load and convert image to RGB."""
    return Image.open(path).convert('RGB')


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load CSV data into DataFrame."""
    return pd.read_csv(filepath)


def create_image_transform_pipeline(input_size: int = 224, 
                                   is_training: bool = True) -> Callable:
    """Create image transformation pipeline."""
    from torchvision import transforms
    
    base_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ]
    
    if is_training:
        base_transforms.insert(1, transforms.RandomHorizontalFlip())
        base_transforms.insert(1, transforms.RandomRotation(10))
        
    return transforms.Compose(base_transforms)


def tokenize_text(text: str, tokenizer=None) -> List[str]:
    """Tokenize text using NLTK or HuggingFace tokenizer."""
    if tokenizer is None:
        return nltk.word_tokenize(text.lower())
    else:
        return tokenizer(text)


class FlexibleImageDataset(Dataset):
    """Flexible image dataset that handles various formats."""
    
    def __init__(self, 
                 data_path: str,
                 label_file: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 image_column: str = 'image',
                 label_column: str = 'label'):
        
        self.data_path = Path(data_path)
        self.transform = transform or create_image_transform_pipeline()
        
        if label_file:
            self.labels_df = load_csv_data(label_file)
            self.image_paths = [
                self.data_path / img for img in self.labels_df[image_column]
            ]
            self.labels = self.labels_df[label_column].tolist()
        else:
            # Auto-discover from folder structure
            self.image_paths, self.labels = self._discover_images()
    
    def _discover_images(self) -> Tuple[List[Path], List[int]]:
        """Auto-discover images from folder structure."""
        image_paths, labels = [], []
        class_folders = sorted([d for d in self.data_path.iterdir() if d.is_dir()])
        
        for label, class_folder in enumerate(class_folders):
            for img_path in class_folder.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_paths.append(img_path)
                    labels.append(label)
                    
        return image_paths, labels
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = load_image(str(self.image_paths[idx]))
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class FlexibleTextDataset(Dataset):
    """Flexible text dataset for classification/regression."""
    
    def __init__(self,
                 data_file: str,
                 text_column: str = 'text',
                 label_column: str = 'label',
                 tokenizer_name: Optional[str] = None,
                 max_length: int = 512):
        
        self.data = load_csv_data(data_file)
        self.texts = self.data[text_column].tolist()
        self.labels = self.data[label_column].tolist()
        self.tokenizer = nltk.word_tokenize
        self.max_length = max_length
        self.vocab = self._build_vocab()
        self.word_to_idx = {word: idx + 1 for idx, word in enumerate(self.vocab)}
        self.word_to_idx['<pad>'] = 0
        self.word_to_idx['<unk>'] = len(self.word_to_idx)

    def _build_vocab(self) -> List[str]:
        all_tokens = []
        for text in self.texts:
            all_tokens.extend(self.tokenizer(str(text).lower()))
        return sorted(list(set(all_tokens)))

    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        tokenized_text = self.tokenizer(text.lower())
        indexed_text = [self.word_to_idx.get(word, self.word_to_idx['<unk>']) for word in tokenized_text]

        if len(indexed_text) < self.max_length:
            indexed_text += [self.word_to_idx['<pad>']] * (self.max_length - len(indexed_text))
        else:
            indexed_text = indexed_text[:self.max_length]
        
        attention_mask = [1 if token_idx != self.word_to_idx['<pad>'] else 0 for token_idx in indexed_text]

        return {
            'input_ids': torch.tensor(indexed_text, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


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
            feature_columns = [col for col in self.data.columns 
                             if col != target_column]
        
        self.features = self.data[feature_columns].values.astype(np.float32)
        self.targets = self.data[target_column].values
        
        if normalize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            self.features = scaler.fit_transform(self.features)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


def create_dataset(data_type: str, **kwargs) -> Dataset:
    """Factory function to create datasets based on data type."""
    dataset_map = {
        'image': FlexibleImageDataset,
        'text': FlexibleTextDataset,
        'tabular': TabularDataset
    }
    
    if data_type not in dataset_map:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    return dataset_map[data_type](**kwargs) 

def get_sample_batch(dataloader: torch.utils.data.DataLoader) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Get a single batch of data from a DataLoader."""
    try:
        sample = next(iter(dataloader))
        # The structure of 'sample' can vary. Common cases:
        # 1. (inputs, labels) tuple
        # 2. inputs tensor (if no labels, e.g., unsupervised VAE)
        # 3. Dictionary for text data: {'input_ids': ..., 'attention_mask': ..., 'label': ...}
        
        if isinstance(sample, list) and len(sample) == 2: # Common case: (inputs, labels)
            inputs, labels = sample
            return inputs, labels
        elif isinstance(sample, torch.Tensor): # Case: only inputs (unsupervised)
            return sample, None
        elif isinstance(sample, dict) and 'input_ids' in sample: # Text data
            # For text, a bit more complex to return a single "input" tensor for general visualization
            # We can return the dict itself, or try to extract primary input
            # For now, let's just return the first tensor found (e.g., input_ids) and labels if present
            inputs = sample['input_ids']
            labels = sample.get('label') #.get because label might not always be there
            return inputs, labels
        else:
            print(f"Warning: Unknown sample structure in get_sample_batch: {type(sample)}")
            return None, None
            
    except StopIteration:
        print("Warning: DataLoader is empty, cannot get a sample batch.")
        return None, None
    except Exception as e:
        print(f"Error getting sample batch: {e}")
        return None, None 