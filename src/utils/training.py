"""
Training utilities with functional programming approach.
Includes trainers, optimizers, schedulers, and evaluation metrics.
"""

from typing import Dict, List, Optional, Callable, Tuple, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_optimizer(model: nn.Module, 
                    optimizer_type: str = 'adam',
                    learning_rate: float = 1e-3,
                    weight_decay: float = 1e-4,
                    **kwargs) -> optim.Optimizer:
    """Create optimizer with specified parameters."""
    optimizer_map = {
        'adam': lambda: optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=weight_decay, **kwargs),
        'sgd': lambda: optim.SGD(model.parameters(), lr=learning_rate, 
                               weight_decay=weight_decay, momentum=0.9, **kwargs),
        'adamw': lambda: optim.AdamW(model.parameters(), lr=learning_rate, 
                                   weight_decay=weight_decay, **kwargs),
        'rmsprop': lambda: optim.RMSprop(model.parameters(), lr=learning_rate, 
                                       weight_decay=weight_decay, **kwargs)
    }
    
    if optimizer_type not in optimizer_map:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    return optimizer_map[optimizer_type]()


def create_scheduler(optimizer: optim.Optimizer,
                    scheduler_type: str = 'step',
                    **kwargs) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    scheduler_map = {
        'step': lambda: optim.lr_scheduler.StepLR(optimizer, **kwargs),
        'cosine': lambda: optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs),
        'plateau': lambda: optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs),
        'exponential': lambda: optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    }
    
    if scheduler_type not in scheduler_map:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    return scheduler_map[scheduler_type]()


def compute_metrics(predictions: torch.Tensor, 
                   targets: torch.Tensor,
                   task_type: str = 'classification') -> Dict[str, float]:
    """Compute evaluation metrics."""
    if task_type == 'classification':
        pred_labels = torch.argmax(predictions, dim=1).cpu().numpy()
        true_labels = targets.cpu().numpy()
        
        return {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'f1': f1_score(true_labels, pred_labels, average='weighted'),
            'precision': precision_score(true_labels, pred_labels, average='weighted'),
            'recall': recall_score(true_labels, pred_labels, average='weighted')
        }
    elif task_type == 'regression':
        mse = nn.MSELoss()(predictions, targets).item()
        mae = nn.L1Loss()(predictions, targets).item()
        return {'mse': mse, 'mae': mae}
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() 
                                   for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


def train_epoch(model: nn.Module,
               train_loader: DataLoader,
               criterion: nn.Module,
               optimizer: optim.Optimizer,
               device: torch.device,
               task_type: str = 'classification') -> Tuple[float, Dict[str, float]]:
    """Train model for one epoch. Simplified for MLP tabular data."""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # For tabular data, batch is (inputs, targets)
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.append(outputs.detach())
        all_targets.append(targets.detach())
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(train_loader)
    
    # Compute metrics
    metrics = {}
    if all_predictions and all_targets:
        predictions_cat = torch.cat(all_predictions)
        targets_cat = torch.cat(all_targets)
        try:
            metrics.update(compute_metrics(predictions_cat, targets_cat, task_type=task_type))
        except Exception as e:
            print(f"Note: Could not compute training metrics: {e}")

    return avg_loss, metrics


def evaluate_model(model: nn.Module,
                  val_loader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device,
                  task_type: str = 'classification') -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation set. Simplified for MLP tabular data."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(val_loader, desc='Validation')
    with torch.no_grad():
        for batch in progress_bar:
            # For tabular data, batch is (inputs, targets)
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_predictions.append(outputs.detach())
            all_targets.append(targets.detach())
            progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(val_loader)

    # Compute metrics
    metrics = {}
    if all_predictions and all_targets:
        predictions_cat = torch.cat(all_predictions)
        targets_cat = torch.cat(all_targets)
        try:
            metrics.update(compute_metrics(predictions_cat, targets_cat, task_type=task_type))
        except Exception as e:
            print(f"Note: Could not compute validation metrics: {e}")
            
    return avg_loss, metrics


def train_model(model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               num_epochs: int = 100,
               optimizer_config: Dict[str, Any] = None,
               scheduler_config: Dict[str, Any] = None,
               early_stopping_config: Dict[str, Any] = None,
               device: torch.device = None,
               data_type: str = 'tabular',  # Always tabular for MLP
               task_type: str = 'classification',
               model_type: str = 'mlp',  # Always mlp
               config: Dict[str, Any] = None) -> Dict[str, List[float]]:
    """Complete training pipeline for MLP models on tabular data."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    optimizer_config = optimizer_config or {'optimizer_type': 'adam', 'learning_rate': 1e-3}
    optimizer = create_optimizer(model, **optimizer_config)
    
    scheduler = None
    if scheduler_config:
        scheduler = create_scheduler(optimizer, **scheduler_config)
    
    early_stopping = None
    if early_stopping_config:
        early_stopping = EarlyStopping(**early_stopping_config)
    
    # Set up criterion and history keys based on task type
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
        history_keys = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'train_f1', 'val_f1']
    elif task_type == 'regression':
        criterion = nn.MSELoss()
        history_keys = ['train_loss', 'val_loss', 'train_mse', 'val_mse', 'train_mae', 'val_mae']
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    history = {key: [] for key in history_keys}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, task_type
        )
        val_loss, val_metrics = evaluate_model(
            model, val_loader, criterion, device, task_type
        )
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if task_type == 'classification':
            history['train_accuracy'].append(train_metrics.get('accuracy', 0))
            history['val_accuracy'].append(val_metrics.get('accuracy', 0))
            history['train_f1'].append(train_metrics.get('f1', 0))
            history['val_f1'].append(val_metrics.get('f1', 0))
            print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_metrics.get('accuracy', 0):.4f}, F1: {train_metrics.get('f1', 0):.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_metrics.get('accuracy', 0):.4f}, F1: {val_metrics.get('f1', 0):.4f}")
        elif task_type == 'regression':
            history['train_mse'].append(train_metrics.get('mse', 0))
            history['val_mse'].append(val_metrics.get('mse', 0))
            history['train_mae'].append(train_metrics.get('mae', 0))
            history['val_mae'].append(val_metrics.get('mae', 0))
            print(f"  Train Loss: {train_loss:.4f} (MSE: {train_metrics.get('mse', 0):.4f}, MAE: {train_metrics.get('mae', 0):.4f})")
            print(f"  Val Loss: {val_loss:.4f} (MSE: {val_metrics.get('mse', 0):.4f}, MAE: {val_metrics.get('mae', 0):.4f})")

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        if early_stopping and early_stopping(val_loss, model):
            print("Early stopping triggered.")
            break
            
    return history


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         model_type: str = 'mlp',
                         task_type: str = 'classification') -> None:
    """Plot training and validation history for MLP models."""
    
    if task_type == 'classification':
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(history.get('train_accuracy', []), label='Train Accuracy')
        axes[1].plot(history.get('val_accuracy', []), label='Validation Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

    elif task_type == 'regression':
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        axes[0].plot(history['train_loss'], label='Train Loss (MSE)')
        axes[0].plot(history['val_loss'], label='Validation Loss (MSE)')
        axes[0].set_title('Total Loss (MSE)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history.get('train_mse', []), label='Train MSE')
        axes[1].plot(history.get('val_mse', []), label='Validation MSE')
        axes[1].set_title('Mean Squared Error (MSE)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(history.get('train_mae', []), label='Train MAE')
        axes[2].plot(history.get('val_mae', []), label='Validation MAE')
        axes[2].set_title('Mean Absolute Error (MAE)')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('MAE')
        axes[2].legend()
        axes[2].grid(True)
    else:
        print(f"Plotting not specifically implemented for task_type: {task_type}, using default loss plot.")
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    plt.show()


def predict_test_set(model: nn.Module,
                    test_loader: DataLoader,
                    device: torch.device,
                    data_type: str = 'tabular',  # Always tabular for MLP
                    model_type: str = 'mlp',  # Always mlp
                    config: Dict[str, Any] = None) -> torch.Tensor:
    """Generate predictions for the test set. Simplified for MLP tabular data."""
    model.eval()
    all_outputs = []
    
    progress_bar = tqdm(test_loader, desc='Predicting')
    with torch.no_grad():
        for batch in progress_bar:
            # For tabular data, batch is (inputs, targets) or just (inputs,)
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                inputs = batch[0]  # Take first element (inputs)
            else:
                inputs = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.cpu())
            
    return torch.cat(all_outputs)


def save_predictions(predictions: torch.Tensor,
                    output_file: str,
                    task_type: str = 'classification') -> None:
    """Save predictions to file."""
    if task_type == 'classification':
        pred_labels = torch.argmax(predictions, dim=1).numpy()
        np.savetxt(output_file, pred_labels, fmt='%d')
    else:  # regression
        pred_values = predictions.squeeze().numpy()
        np.savetxt(output_file, pred_values, fmt='%.6f')
    
    print(f'Predictions saved to {output_file}') 