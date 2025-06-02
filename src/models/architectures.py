"""
Flexible model architectures for different data types and tasks.
Implements transfer learning and custom architectures.
"""

from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .vae_model import VAE, vae_loss_function
from .cyclegan_model import ResNetGenerator, NLayerDiscriminator


def create_classifier_head(input_dim: int, 
                          num_classes: int, 
                          hidden_dims: Optional[list] = None,
                          dropout: float = 0.5) -> nn.Module:
    """Create a flexible classifier head."""
    if hidden_dims is None:
        hidden_dims = [256, 128]
    
    layers = []
    current_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        current_dim = hidden_dim
    
    layers.append(nn.Linear(current_dim, num_classes))
    return nn.Sequential(*layers)


def freeze_model_layers(model: nn.Module, freeze_until: Optional[str] = None) -> None:
    """Freeze model layers up to a specified layer name."""
    freeze_all = freeze_until is None
    
    for name, param in model.named_parameters():
        if freeze_all or freeze_until not in name:
            param.requires_grad = False
        else:
            break


class FlexibleImageClassifier(nn.Module):
    """Flexible image classifier with transfer learning."""
    
    def __init__(self, 
                 num_classes: int,
                 backbone: str = 'resnet18',
                 pretrained: bool = True,
                 freeze_backbone: bool = True,
                 classifier_hidden_dims: Optional[list] = None):
        
        super().__init__()
        
        # Load backbone
        self.backbone = self._create_backbone(backbone, pretrained)
        
        # Get feature dimension
        feature_dim = self._get_feature_dimension()
        
        # Replace classifier
        self.backbone.fc = create_classifier_head(
            feature_dim, num_classes, classifier_hidden_dims
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            freeze_model_layers(self.backbone, 'fc')
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Create backbone model."""
        backbone_map = {
            'resnet18': lambda: models.resnet18(pretrained=pretrained),
            'resnet50': lambda: models.resnet50(pretrained=pretrained),
            'efficientnet_b0': lambda: models.efficientnet_b0(pretrained=pretrained),
            'vit_b_16': lambda: models.vit_b_16(pretrained=pretrained),
        }
        
        if backbone not in backbone_map:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return backbone_map[backbone]()
    
    def _get_feature_dimension(self) -> int:
        """Get the feature dimension of the backbone."""
        # Handle different backbone types
        if hasattr(self.backbone, 'fc'):
            return self.backbone.fc.in_features
        elif hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Sequential):
                return self.backbone.classifier[-1].in_features
            else:
                return self.backbone.classifier.in_features
        elif hasattr(self.backbone, 'head'):
            return self.backbone.head.in_features
        else:
            raise ValueError("Cannot determine feature dimension")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SimpleTextClassifier(nn.Module):
    """Simple text classifier with Embedding and LSTM."""
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_lstm_layers: int = 1,
                 bidirectional_lstm: bool = True,
                 dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim,
                              hidden_dim,
                              num_layers=num_lstm_layers,
                              bidirectional=bidirectional_lstm,
                              batch_first=True,
                              dropout=dropout if num_lstm_layers > 1 else 0) # Dropout only if multiple LSTM layers
        
        lstm_output_dim = hidden_dim * 2 if bidirectional_lstm else hidden_dim
        self.classifier = create_classifier_head(lstm_output_dim, num_classes, dropout=dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attention_mask is not directly used by nn.LSTM if batch_first=True and packed sequences are not used.
        # However, it's good practice to accept it if other parts of the pipeline provide it.
        # For this simple model, we rely on padding_idx in nn.Embedding.
        embedded = self.embedding(input_ids)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * num_directions)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)

        # We can use the final hidden state or an aggregation of all hidden states.
        # Here, we concatenate the final hidden states from both directions (if bidirectional).
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            final_hidden_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            final_hidden_state = hidden[-1,:,:]
            
        return self.classifier(final_hidden_state)


class MLPClassifier(nn.Module):
    """Multi-layer perceptron for tabular data."""
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: Optional[list] = None,
                 dropout: float = 0.3):
        
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.classifier = create_classifier_head(
            input_dim, num_classes, hidden_dims, dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AutoEncoder(nn.Module):
    """Simple autoencoder for unsupervised learning."""
    
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def create_model(model_type: str, **kwargs):
    """Create model based on type and config."""
    if model_type == 'image_classifier':
        return FlexibleImageClassifier(**kwargs)
    elif model_type == 'simple_text_classifier':
        # Ensure vocab_size, embedding_dim, hidden_dim are passed via kwargs
        # These would typically come from the dataset preparation or config
        text_model_params = {
            'vocab_size': kwargs.get('vocab_size'),
            'embedding_dim': kwargs.get('embedding_dim', 100),
            'hidden_dim': kwargs.get('hidden_dim', 128),
            'num_classes': kwargs.get('num_classes'),
            'num_lstm_layers': kwargs.get('num_lstm_layers', 1),
            'bidirectional_lstm': kwargs.get('bidirectional_lstm', True),
            'dropout': kwargs.get('dropout', 0.5)
        }
        # Validate required parameters
        if not text_model_params['vocab_size'] or not text_model_params['num_classes']:
            raise ValueError("vocab_size and num_classes must be provided for simple_text_classifier")
        return SimpleTextClassifier(**text_model_params)
    elif model_type == 'mlp':
        return MLPClassifier(**kwargs)
    elif model_type == 'vae':
        vae_params = {
            'input_dim': kwargs.get('input_dim', 28*28),
            'encoder_hidden_dims': kwargs.get('encoder_hidden_dims', [512, 256]),
            'latent_dim': kwargs.get('latent_dim', 64),
            'decoder_hidden_dims': kwargs.get('decoder_hidden_dims', [256, 512]),
            'image_dims': kwargs.get('image_dims', (1,28,28))
        }
        return VAE(**vae_params)
    elif model_type == 'cyclegan_generator':
        gen_params = {
            'input_nc': kwargs.get('input_nc', 3),
            'output_nc': kwargs.get('output_nc', 3),
            'ngf': kwargs.get('ngf', 64),
            'n_blocks': kwargs.get('n_blocks_generator', 9)
        }
        return ResNetGenerator(**gen_params)
    elif model_type == 'cyclegan_discriminator':
        disc_params = {
            'input_nc': kwargs.get('input_nc', 3),
            'ndf': kwargs.get('ndf', 64),
            'n_layers': kwargs.get('n_layers_discriminator', 3)
        }
        return NLayerDiscriminator(**disc_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def apply_weight_init(model: nn.Module, init_type: str = 'xavier') -> None:
    """Apply weight initialization to model."""
    init_map = {
        'xavier': nn.init.xavier_uniform_,
        'kaiming': nn.init.kaiming_uniform_,
        'normal': lambda x: nn.init.normal_(x, 0, 0.02)
    }
    
    init_fn = init_map.get(init_type, nn.init.xavier_uniform_)
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init_fn(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0) 