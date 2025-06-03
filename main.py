#!/usr/bin/env python3
"""
Main entry point for the MLP deep learning framework for tabular data.
Provides a pipeline for data analysis, model training, and prediction.
"""

import argparse
import json
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from src.data.dataset import create_dataset, get_sample_batch
from src.models.architectures import create_model
from src.utils.data_utils import (
    load_and_analyze_data, 
    get_recommended_config,
    create_data_splits,
    create_data_loaders,
    visualize_data_distribution,
    create_submission_file
)
from src.utils.training import (
    train_model,
    predict_test_set,
    save_predictions,
    plot_training_history
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MLP Deep Learning Framework for Tabular Data'
    )
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the tabular dataset CSV file')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration JSON file for MLP task')
    
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze data without training')
    
    parser.add_argument('--predict_only', action='store_true',
                       help='Only predict using pre-trained model')
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pre-trained model for prediction')
    
    parser.add_argument('--test_path', type=str, default=None, 
                       help='Path to test data CSV for prediction')
    
    parser.add_argument('--submission_format', type=str, 
                       choices=['simple', 'kaggle', 'indexed'], 
                       default='simple',
                       help='Submission file format')
    
    return parser.parse_args()


def load_config(config_path: str, data_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load configuration from file. 
       Data analysis based recommendation is less critical for a focused MLP setup but kept for now.
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Ensure essential keys for MLP are present or provide defaults
    config.setdefault('data_type', 'tabular')
    config.setdefault('model_type', 'mlp')
    config.setdefault('batch_size', 32)
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('num_epochs', 50)
    config.setdefault('task_type', 'classification') # 'classification' or 'regression'
    
    if config['data_type'] != 'tabular' or config['model_type'] != 'mlp':
        print(f"Warning: Config data_type is {config['data_type']} and model_type is {config['model_type']}. Framework is focused on tabular MLP.")
        # Force to tabular MLP if not specified, or raise error
        # For now, we'll let it proceed but MLP model creation will fail if input_dim isn't right.

    return config


def prepare_datasets(data_path: str, config: Dict[str, Any]):
    """Prepare datasets based on configuration. Focused on tabular data."""
    if config['data_type'] != 'tabular':
        raise ValueError(f"Unsupported data_type: {config['data_type']}. Only 'tabular' is supported.")
    
    dataset = create_dataset(
        data_type='tabular',
        data_file=data_path,
        target_column=config.get('target_column', 'target'),
        feature_columns=config.get('feature_columns'), # Optional, TabularDataset can infer
        normalize=config.get('normalize', True)
    )
    return dataset


def create_model_from_config(config: Dict[str, Any], num_classes: int, input_dim: int):
    """Create model based on configuration. Focused on MLP."""
    if config['model_type'] != 'mlp':
        raise ValueError(f"Unsupported model_type: {config['model_type']}. Only 'mlp' is supported.")
    
    model_creation_params = {
        'input_dim': input_dim,
        'num_classes': num_classes, # For classification, for regression this is used by head if not 1
        'hidden_dims': config.get('hidden_dims', [128, 64]),
        'dropout': config.get('dropout', 0.3),
        'task_type': config.get('task_type', 'classification')
    }
    return create_model(model_type='mlp', **model_creation_params)


def run_analysis_phase(data_path: str, config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Run data analysis phase for tabular data."""
    print("📊 Analyzing dataset...")
    # load_and_analyze_data might need to be told it's tabular if it was more general before
    # For now, assume it works or will be adapted if it relied on data_type in the config too much
    data_analysis = load_and_analyze_data(data_path, data_type='tabular', target_column=config.get('target_column'))
    
    analysis_file = output_dir / 'data_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(data_analysis, f, indent=2)
    
    print("Data Analysis Results:")
    # Adjust prints for tabular specifics
    print(f"  Data Type: {data_analysis.get('data_type', 'tabular')}")
    if 'num_classes' in data_analysis and config.get('task_type') == 'classification':
        print(f"  Number of Classes: {data_analysis['num_classes']}")
    if 'num_features' in data_analysis:
        print(f"  Number of Features: {data_analysis['num_features']}")
    print(f"  Total Samples: {data_analysis.get('num_samples', 'Unknown')}")
    
    if config.get('task_type') == 'classification':
        print(f"  Balanced: {data_analysis.get('balanced', 'Unknown')}") # If still relevant
    
    try:
        # visualize_data_distribution might need target_column from config
        visualize_data_distribution(data_analysis, str(output_dir / 'data_distribution.png'), target_column=config.get('target_column'))
    except Exception as e:
        print(f"Warning: Could not create data distribution visualization: {e}")
    
    return data_analysis


def run_training_phase(data_path: str, config: Dict[str, Any], 
                      data_analysis: Dict[str, Any], output_dir: Path):
    """Run training phase for MLP model."""
    print("🏋️ Starting training phase...")
    
    dataset = prepare_datasets(data_path, config) # dataset is TabularDataset
    
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset, 
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.15)
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 0) # Default to 0 for simpler setup
    )
    
    # Determine num_classes and input_dim
    input_dim = dataset.num_features # Get from TabularDataset
    
    if config.get('task_type') == 'regression':
        num_classes = 1 # For regression, the model head outputs 1 value
    else: # classification
        # Infer num_classes from dataset if possible, or use config, or from analysis
        if hasattr(dataset, 'targets') and not pd.api.types.is_float_dtype(dataset.targets):
             num_classes = len(torch.unique(torch.tensor(dataset.targets)))
        else: # Fallback if targets are float (regression) or not easily determined here
            num_classes = data_analysis.get('num_classes', config.get('num_classes', 2)) # Default to 2 for classification
            if num_classes > 200 and config.get('task_type') == 'classification': # Heuristic for large number of classes
                print(f"Warning: Large number of classes ({num_classes}) detected. Ensure this is correct.")
    
    model = create_model_from_config(config, num_classes, input_dim)
    
    print(f"Model created: {config['model_type']} for {config.get('task_type', 'classification')}")
    print(f"  Input features: {input_dim}, Output units: {num_classes if config.get('task_type') == 'classification' else 1}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_epochs', 50),
        optimizer_config=config.get('optimizer_config', {'optimizer_type': 'adam', 'learning_rate': config.get('learning_rate', 1e-3)}),
        scheduler_config=config.get('scheduler_config'),
        early_stopping_config=config.get('early_stopping'),
        data_type='tabular', # Hardcoded as this is the focus
        task_type=config.get('task_type', 'classification'),
        model_type='mlp', # Hardcoded
        config=config,
        device=device # Pass device to train_model
    )
    
    model_path = output_dir / 'best_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    plot_training_history(history, 
                          str(output_dir / 'training_history.png'),
                          model_type='mlp', # Hardcoded
                          task_type=config.get('task_type', 'classification')
                          )
    
    history_file = output_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    return model, test_loader


def run_prediction_phase(model, test_loader, config: Dict[str, Any], 
                        output_dir: Path, submission_format: str):
    """Run prediction phase for MLP model."""
    print("🔮 Generating predictions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device) # Ensure model is on the correct device
    
    predictions = predict_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        data_type='tabular', # Hardcoded
        model_type='mlp', # Hardcoded
        config=config # Pass full config for task_type etc.
    )
    
    task_type = config.get('task_type', 'classification')
    
    if task_type == 'classification':
        # For classification, predictions are raw logits/probabilities
        if predictions.ndim > 1 and predictions.shape[1] > 1: # Logits for multi-class
            pred_outputs = torch.argmax(predictions, dim=1).cpu().numpy()
        else: # Binary classification (single output neuron with sigmoid) or already processed
            pred_outputs = (predictions.squeeze().cpu().numpy() > 0.5).astype(int) if predictions.shape[1] == 1 else predictions.squeeze().cpu().numpy()
    else: # Regression
        pred_outputs = predictions.squeeze().cpu().numpy() # Ensure it's squeezed to 1D array
        
    create_submission_file(
        pred_outputs, 
        submission_format=submission_format,
        output_file=str(output_dir / 'predictions.csv')
    )
    
    print(f"Predictions saved to {output_dir / 'predictions.csv'}")


def main():
    """Main function for MLP framework."""
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure parent dirs are created
    
    print("🚀 Starting MLP Deep Learning Framework")
    print(f"Data path: {args.data_path}")
    print(f"Config path: {args.config}")
    print(f"Output directory: {output_dir}")
    
    # Load configuration first
    # data_analysis for config recommendation is removed for simplicity, config is now required
    config = load_config(args.config)
    
    # Save final config to output_dir for reproducibility
    final_config_path = output_dir / 'final_run_config.json'
    with open(final_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Final configuration saved to {final_config_path}")
    
    if not args.predict_only:
        data_analysis = run_analysis_phase(args.data_path, config, output_dir)
    else:
        # For predict_only, we might still need num_classes and input_dim
        # Try to load a previous analysis or infer from a dummy dataset object
        # This part needs care: if we don't run analysis, where do input_dim/num_classes come from for model loading?
        # For now, assume they are in the config or can be inferred.
        # The config might need to store num_features (input_dim) and num_classes from the training run.
        print("Predict-only mode: Skipping data analysis. Ensure config has necessary model parameters.")
        # A minimal data_analysis like object might be needed by some downstream funcs if not careful
        # Example: what if create_model_from_config for loading needs num_classes from analysis?
        # This is handled now by passing input_dim and num_classes to create_model_from_config directly.
        # These must be in the config or determined before model loading in predict_only.
        data_analysis = {} # Placeholder, ideally config should be self-sufficient for predict_only model loading

    if args.analyze_only:
        print("✅ Analysis complete!")
        return
    
    # Phase 3: Training or Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.predict_only:
        # Removed CycleGAN specific logic
        model, test_loader = run_training_phase(
            args.data_path, config, data_analysis, output_dir
        )
    else: # predict_only mode
        if not args.model_path:
            raise ValueError("Model path (--model_path) required for prediction-only mode.")
        if not args.test_path:
            raise ValueError("Test data path (--test_path) required for prediction-only mode.")

        # For predict_only, we need input_dim and num_classes to recreate model structure
        # These should ideally be in the saved config or determinable
        # For now, we'll expect them in the main config used for prediction
        if 'input_dim' not in config or ('num_classes' not in config and config.get('task_type') !='regression'):
             raise ValueError("Config for predict_only must contain 'input_dim' and 'num_classes' (for classification).")

        input_dim_pred = config['input_dim']
        num_classes_pred = config.get('num_classes', 1) if config.get('task_type') == 'regression' else config['num_classes']

        model = create_model_from_config(config, num_classes_pred, input_dim_pred)
        
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        
        print(f"Loading test data from {args.test_path}")
        # Test dataset config should align with training data config (e.g. feature columns, normalization)
        # We use the main config for preparing test dataset.
        test_dataset_pred = prepare_datasets(args.test_path, config)
        
        # Check consistency of features if possible (e.g. if train_dataset was available or info stored)
        if hasattr(test_dataset_pred, 'num_features') and test_dataset_pred.num_features != input_dim_pred:
            print(f"Warning: Test data has {test_dataset_pred.num_features} features, but model expects {input_dim_pred}.")
            # Potentially raise error or try to adapt if feature names are available and can be matched

        from torch.utils.data import DataLoader # Keep local import if not widely used
        test_loader = DataLoader(test_dataset_pred, batch_size=config.get('batch_size', 32), shuffle=False, num_workers=config.get('num_workers',0))
    
    # Phase 4: Prediction
    if 'model' in locals() and 'test_loader' in locals():
         run_prediction_phase(
             model, test_loader, config, output_dir, args.submission_format
         )
    else:
         print("Warning: Model and/or test_loader not available for prediction phase. Skipping.")

    print("🎉 MLP Pipeline complete!")
    print(f"Check {output_dir} for all outputs")


if __name__ == "__main__":
    main() 