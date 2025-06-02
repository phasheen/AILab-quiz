#!/usr/bin/env python3
"""
Main entry point for the flexible deep learning quiz framework.
Provides a complete pipeline for data analysis, model training, and prediction.
"""

import argparse
import json
import torch
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
from src.utils.cyclegan_training import train_cyclegan
from src.utils.visualization_utils import (
    visualize_reconstructions,
    plot_latent_space_distribution,
    visualize_generated_images
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Flexible Deep Learning Quiz Framework'
    )
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the dataset directory')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration JSON file')
    
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze data without training')
    
    parser.add_argument('--predict_only', action='store_true',
                       help='Only predict using pre-trained model')
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pre-trained model for prediction')
    
    parser.add_argument('--test_path', type=str, default=None,
                       help='Path to test data for prediction')
    
    parser.add_argument('--submission_format', type=str, 
                       choices=['simple', 'kaggle', 'indexed'], 
                       default='simple',
                       help='Submission file format')
    
    return parser.parse_args()


def load_config(config_path: str = None, data_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load configuration from file or generate from data analysis."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    elif data_analysis:
        return get_recommended_config(data_analysis)
    else:
        # Default configuration
        return {
            'data_type': 'image',
            'model_type': 'image_classifier',
            'batch_size': 32,
            'learning_rate': 1e-3,
            'num_epochs': 50
        }


def prepare_datasets(data_path: str, config: Dict[str, Any]):
    """Prepare datasets based on configuration."""
    data_type = config['data_type']
    
    if data_type == 'image':
        dataset = create_dataset(
            data_type='image',
            data_path=data_path,
            label_file=config.get('label_file'),
            image_column=config.get('image_column', 'image'),
            label_column=config.get('label_column', 'label')
        )
    
    elif data_type == 'text':
        dataset = create_dataset(
            data_type='text',
            data_file=data_path,
            text_column=config.get('text_column', 'text'),
            label_column=config.get('label_column', 'label'),
            max_length=config.get('max_length', 512)
        )
    
    elif data_type == 'tabular':
        dataset = create_dataset(
            data_type='tabular',
            data_file=data_path,
            target_column=config.get('target_column', 'target'),
            feature_columns=config.get('feature_columns'),
            normalize=config.get('normalize', True)
        )
    
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    return dataset


def create_model_from_config(config: Dict[str, Any], num_classes: int, dataset: Optional[Any] = None):
    """Create model based on configuration."""
    model_type = config['model_type']
    
    # Consolidate model creation logic into src.models.architectures.create_model
    # Pass all necessary parameters from config. The create_model function in architectures.py
    # will select what it needs based on the model_type.
    
    model_creation_params = {
        'num_classes': num_classes, # For classifiers
        **config # Pass the entire config dictionary
    }

    if model_type == 'simple_text_classifier':
        if dataset is None or not hasattr(dataset, 'word_to_idx'):
            raise ValueError("Dataset with word_to_idx is required for simple_text_classifier")
        model_creation_params['vocab_size'] = len(dataset.word_to_idx)

    return create_model(model_type=model_type, **model_creation_params)


def run_analysis_phase(data_path: str, output_dir: Path) -> Dict[str, Any]:
    """Run data analysis phase."""
    print("📊 Analyzing dataset...")
    data_analysis = load_and_analyze_data(data_path)
    
    # Save analysis results
    analysis_file = output_dir / 'data_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(data_analysis, f, indent=2)
    
    print(f"Data Analysis Results:")
    print(f"  Data Type: {data_analysis['data_type']}")
    print(f"  Number of Classes: {data_analysis.get('num_classes', 'Unknown')}")
    print(f"  Total Samples: {data_analysis.get('total_images') or data_analysis.get('num_samples', 'Unknown')}")
    print(f"  Balanced: {data_analysis.get('balanced', 'Unknown')}")
    
    # Create visualizations
    try:
        visualize_data_distribution(data_analysis, str(output_dir / 'data_distribution.png'))
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")
    
    return data_analysis


def run_training_phase(data_path: str, config: Dict[str, Any], 
                      data_analysis: Dict[str, Any], output_dir: Path):
    """Run training phase."""
    print("🏋️ Starting training phase...")
    
    # Prepare datasets
    dataset = prepare_datasets(data_path, config)
    
    # Create splits
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset, 
        train_ratio=config.get('train_ratio', 0.7),
        val_ratio=config.get('val_ratio', 0.15),
        test_ratio=config.get('test_ratio', 0.15)
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4)
    )
    
    # Create model
    num_classes = data_analysis.get('num_classes', 2)
    model = create_model_from_config(config, num_classes, dataset=dataset)
    
    print(f"Model created: {config['model_type']}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_epochs', 50),
        optimizer_config=config.get('optimizer_config'),
        scheduler_config=config.get('scheduler_config'),
        early_stopping_config=config.get('early_stopping'),
        data_type=config['data_type'],
        task_type=config.get('task_type', 'classification'),
        model_type=config['model_type'],
        config=config
    )
    
    # Save model
    model_path = output_dir / 'best_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    plot_training_history(history, 
                          str(output_dir / 'training_history.png'),
                          model_type=config['model_type'],
                          task_type=config.get('task_type', 'classification')
                          )
    
    # Save training history
    history_file = output_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    # Additional VAE-specific visualizations after training
    if config['model_type'] == 'vae':
        print("Generating VAE specific visualizations...")
        model.eval()
        sample_batch, _ = get_sample_batch(val_loader) # Get a sample batch from validation set
        if sample_batch is not None:
            sample_batch = sample_batch.to(next(model.parameters()).device)
            reconstructions, mu, _ = model(sample_batch)
            visualize_reconstructions(
                sample_batch,
                reconstructions,
                n=8,
                save_path=str(output_dir / 'vae_reconstructions.png')
            )
            plot_latent_space_distribution(
                mu,
                # labels=sample_labels, # If val_loader provides labels and you want to color by them
                title="VAE Latent Space (Validation Samples)",
                save_path=str(output_dir / 'vae_latent_space.png')
            )
        else:
            print("Could not get sample batch for VAE visualizations.")

    return model, test_loader


def run_prediction_phase(model, test_loader, config: Dict[str, Any], 
                        output_dir: Path, submission_format: str):
    """Run prediction phase."""
    print("🔮 Generating predictions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate predictions
    predictions = predict_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        data_type=config['data_type'],
        model_type=config['model_type'],
        config=config
    )
    
    # Save predictions
    task_type = config.get('task_type', 'classification')
    
    if task_type == 'classification':
        pred_labels = torch.argmax(predictions, dim=1).numpy()
        create_submission_file(
            pred_labels, 
            submission_format=submission_format,
            output_file=str(output_dir / 'predictions.csv')
        )
    else:
        pred_values = predictions.squeeze().numpy()
        create_submission_file(
            pred_values,
            submission_format=submission_format, 
            output_file=str(output_dir / 'predictions.csv')
        )
    
    print(f"Predictions saved to {output_dir / 'predictions.csv'}")


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting Flexible Deep Learning Quiz Framework")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {output_dir}")
    
    # Phase 1: Data Analysis
    if not args.predict_only:
        data_analysis = run_analysis_phase(args.data_path, output_dir)
    else:
        data_analysis = {'data_type': 'unknown'}
    
    if args.analyze_only:
        print("✅ Analysis complete!")
        return
    
    # Phase 2: Configuration
    config = load_config(args.config, data_analysis)
    
    # Save final config
    config_file = output_dir / 'final_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration: {config['model_type']} for {config['data_type']} data")
    
    # Phase 3: Training or Load Model
    if not args.predict_only:
        # Special handling for CycleGAN training as it's different
        if config['model_type'].startswith('cyclegan'):
            print("🚲 Starting CycleGAN training mode...")
            
            common_nc = config.get('channels', 3)
            # We need to ensure that the main config dict passed to create_model_from_config
            # contains all necessary sub-keys that create_model in architectures.py will use.
            # For CycleGAN, these keys are 'input_nc', 'output_nc', 'ngf', 'n_blocks_generator', 'ndf', 'n_layers_discriminator'.
            # We can prepare specialized config dicts for each model creation call.

            gen_config_A2B = {**config, 'model_type': 'cyclegan_generator', 'input_nc': common_nc, 'output_nc': common_nc}
            gen_config_B2A = {**config, 'model_type': 'cyclegan_generator', 'input_nc': common_nc, 'output_nc': common_nc}
            disc_config_A = {**config, 'model_type': 'cyclegan_discriminator', 'input_nc': common_nc}
            disc_config_B = {**config, 'model_type': 'cyclegan_discriminator', 'input_nc': common_nc}

            # num_classes is not strictly needed for GANs, pass 0 or a default.
            # The create_model in architectures.py should ideally ignore it for GAN types.
            netG_A2B = create_model_from_config(gen_config_A2B, num_classes=0) 
            netG_B2A = create_model_from_config(gen_config_B2A, num_classes=0)
            netD_A = create_model_from_config(disc_config_A, num_classes=0)
            netD_B = create_model_from_config(disc_config_B, num_classes=0)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            netG_A2B.to(device)
            netG_B2A.to(device)
            netD_A.to(device)
            netD_B.to(device)

            # DataLoaders for two domains (e.g., horses and zebras)
            # This requires dataset handling for two separate (unpaired) image folders
            # Current prepare_datasets is for single dataset. Needs enhancement or specific CycleGAN data prep.
            # For now, let's assume data_path points to a root with domainA and domainB subfolders
            path_A = Path(args.data_path) / config.get('domain_A_folder', 'trainA')
            path_B = Path(args.data_path) / config.get('domain_B_folder', 'trainB')
            
            dataset_A = prepare_datasets(str(path_A), {**config, 'data_type': 'image', 'label_file': None}) # Unsupervised
            dataset_B = prepare_datasets(str(path_B), {**config, 'data_type': 'image', 'label_file': None})
            
            # Ensure create_data_splits and create_data_loaders can handle unsupervised (no labels) or adapt
            # For CycleGAN, we typically don't have val/test splits in the same way for the GAN training part itself.
            # We use the full datasets for training the image translation.
            # However, our create_data_loaders expects train/val/test datasets. We can pass None for val/test.

            # _, _, test_dataset = create_data_splits(...) # Not standard for GAN training itself
            # We need simple DataLoaders from dataset_A and dataset_B
            from torch.utils.data import DataLoader # Direct import for simplicity here
            dataloader_A = DataLoader(dataset_A, batch_size=config.get('batch_size', 1), 
                                      shuffle=True, num_workers=config.get('num_workers', 2))
            dataloader_B = DataLoader(dataset_B, batch_size=config.get('batch_size', 1), 
                                      shuffle=True, num_workers=config.get('num_workers', 2))

            history = train_cyclegan(
                netG_A2B, netG_B2A, netD_A, netD_B,
                dataloader_A, dataloader_B,
                num_epochs=config.get('num_epochs', 200),
                config=config,
                device=device,
                output_dir=output_dir
            )
            # Note: CycleGAN history plotting might need a dedicated function
            # For now, we just save the history dict.
            history_file = output_dir / 'cyclegan_training_history.json'
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"CycleGAN training complete. History saved to {history_file}")
            # No standard model/test_loader to return for the generic prediction phase here.
            # Prediction would be using a specific generator (e.g., netG_A2B) on new data.
        else:
            # Standard training path for other models
            model, test_loader = run_training_phase(
                args.data_path, config, data_analysis, output_dir
            )
    else: # predict_only mode
        if not args.model_path:
            raise ValueError("Model path required for prediction-only mode")
        
        # Load pre-trained model
        model_params = {
            'num_classes': data_analysis.get('num_classes', config.get('num_classes', 2)),
            **config
        }
        model = create_model_from_config(model_params, model_params['num_classes'])
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        
        # Prepare test dataset
        if args.test_path:
            test_dataset = prepare_datasets(args.test_path, config)
            from torch.utils.data import DataLoader
            test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 32))
        else:
            raise ValueError("Test path required for prediction-only mode")
    
    # Phase 4: Prediction
    # Skip standard prediction phase if CycleGAN was trained, as it needs specific handling
    if not (not args.predict_only and config['model_type'].startswith('cyclegan')):
        if args.predict_only and config['model_type'].startswith('cyclegan'):
            print("CycleGAN prediction: Load a generator and run on test data.")
            # This part needs a dedicated prediction script/mode for CycleGAN
            # For example: python main.py --predict_only --model_type cyclegan_generator_A2B --model_path path/to/G_A2B.pth --data_path path/to/domainA_test_images ...
            # For now, this is a placeholder.
            print("CycleGAN prediction not fully implemented in this generic pipeline yet.")
        elif 'model' in locals() and 'test_loader' in locals(): # Ensure model and test_loader exist
             run_prediction_phase(
                 model, test_loader, config, output_dir, args.submission_format
             )
        elif not config['model_type'].startswith('cyclegan'):
             print("Warning: Model and test_loader not available for prediction phase. Skipping.")

    print("🎉 Pipeline complete!")
    print(f"Check {output_dir} for all outputs")


if __name__ == "__main__":
    main() 