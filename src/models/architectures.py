"""
Flexible model architectures for different data types and tasks.
Implements transfer learning and custom architectures.
"""

from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision import models # Removed
# from transformers import AutoModel, AutoConfig # Removed
# from .vae_model import VAE, vae_loss_function # Removed, assuming vae_model.py will be deleted
# from .cyclegan_model import ResNetGenerator, NLayerDiscriminator # Removed, assuming cyclegan_model.py will be deleted


def create_classifier_head(input_dim: int, 
                          num_classes: int, 
                          hidden_dims: Optional[list] = None,
                          dropout: float = 0.5, 
                          is_regression: bool = False) -> nn.Module:
    """Create a flexible classifier head. Can also be used for regression output."""
    if hidden_dims is None:
        hidden_dims = [256, 128]
    
    layers = []
    current_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Consider if BatchNorm is always desired for MLP
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        current_dim = hidden_dim
    
    if is_regression:
        layers.append(nn.Linear(current_dim, 1)) # Output 1 value for regression
    else:
        layers.append(nn.Linear(current_dim, num_classes))
        # Softmax/Sigmoid is usually applied in loss function for classification
        
    return nn.Sequential(*layers)


# def freeze_model_layers(model: nn.Module, freeze_until: Optional[str] = None) -> None: # Potentially unused for MLP only
#     """Freeze model layers up to a specified layer name."""
#     freeze_all = freeze_until is None
#     
#     for name, param in model.named_parameters():
#         if freeze_all or freeze_until not in name:
#             param.requires_grad = False
#         else:
#             break


# class FlexibleImageClassifier(nn.Module): # Removed
# # ... (contents of FlexibleImageClassifier removed)

# class SimpleTextClassifier(nn.Module): # Removed
# # ... (contents of SimpleTextClassifier removed)


class MLPClassifier(nn.Module):
    """Multi-layer perceptron for tabular data."""
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int, # For classification, for regression this will be 1 if not handled by head
                 hidden_dims: Optional[list] = None,
                 dropout: float = 0.3,
                 task_type: str = 'classification'): # Added task_type
        
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.is_regression = task_type == 'regression'
        
        self.classifier_head = create_classifier_head(
            input_dim, 
            num_classes if not self.is_regression else 1, 
            hidden_dims, 
            dropout,
            is_regression=self.is_regression
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(x)


# class AutoEncoder(nn.Module): # Removed
# # ... (contents of AutoEncoder removed)

# Potentially remove vae_model.py and cyclegan_model.py files if they exist and are now orphaned.
# For VAE, the following would be removed from create_model:
# elif model_type == 'vae':
#         vae_params = {
#             'input_dim': kwargs.get('input_dim', 28*28), # This was image specific
#             'encoder_hidden_dims': kwargs.get('encoder_hidden_dims', [512, 256]),
#             'latent_dim': kwargs.get('latent_dim', 64),
#             'decoder_hidden_dims': kwargs.get('decoder_hidden_dims', [256, 512]),
#             'image_dims': kwargs.get('image_dims', (1,28,28)) # Image specific
#         }
#         return VAE(**vae_params)
# For CycleGAN, the following would be removed:
# elif model_type == 'cyclegan_generator':
#         gen_params = { ... }
#         return ResNetGenerator(**gen_params)
#     elif model_type == 'cyclegan_discriminator':
#         disc_params = { ... }
#         return NLayerDiscriminator(**disc_params)

def create_model(model_type: str, **kwargs):
    """Create model based on type and config. Simplified for MLP focus."""
    if model_type == 'mlp':
        # input_dim will be passed from main.py after dataset is prepared
        mlp_params = {
            'input_dim': kwargs.get('input_dim'),
            'num_classes': kwargs.get('num_classes'), 
            'hidden_dims': kwargs.get('hidden_dims'),
            'dropout': kwargs.get('dropout', 0.3),
            'task_type': kwargs.get('task_type', 'classification')
        }
        if not mlp_params['input_dim'] or not mlp_params['num_classes']:
            raise ValueError("input_dim and num_classes must be provided for mlp model")
        return MLPClassifier(**mlp_params)
    else:
        raise ValueError(f"Unsupported model type for MLP focus: {model_type}. Only 'mlp' is supported.")


def apply_weight_init(model: nn.Module, init_type: str = 'xavier') -> None:
    """Apply weight initialization to model."""
    # This function can be kept as it's a general utility
    init_map = {
        'xavier': nn.init.xavier_uniform_,
        'kaiming': nn.init.kaiming_uniform_,
        'normal': lambda x: nn.init.normal_(x, 0, 0.02)
    }
    
    init_fn = init_map.get(init_type, nn.init.xavier_uniform_)
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)): # Conv2d won't be used in MLP
            if isinstance(module, nn.Linear):
                 init_fn(module.weight)
                 if module.bias is not None:
                     nn.init.constant_(module.bias, 0) 