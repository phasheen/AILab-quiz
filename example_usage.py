#!/usr/bin/env python3
"""
Example script showing how to use the MLP Deep Learning Framework programmatically.
This demonstrates the key functions and workflow for tabular data classification.
"""

import torch
import json
from pathlib import Path

# Import the framework components
from src.data.dataset import create_dataset
from src.models.architectures import create_model
from src.utils.data_utils import (
    load_and_analyze_data,
    create_data_splits,
    create_data_loaders
)
from src.utils.training import train_model


def example_classification():
    """Example of training an MLP classifier on the sample data."""
    
    print("🚀 MLP Framework - Programmatic Usage Example")
    print("=" * 50)
    
    # 1. Analyze the data
    print("📊 Step 1: Analyzing dataset...")
    data_path = "sample_data/sample_mlp_data.csv"
    target_column = "target"
    
    data_analysis = load_and_analyze_data(
        data_path=data_path,
        data_type='tabular',
        target_column=target_column
    )
    
    print(f"✅ Found {data_analysis['num_samples']} samples with {data_analysis['num_numeric_features']} features")
    print(f"✅ Task type: {data_analysis['task_type']} with {data_analysis['num_classes']} classes")
    
    # 2. Create dataset
    print("\n📁 Step 2: Creating dataset...")
    dataset = create_dataset(
        data_type='tabular',
        data_file=data_path,
        target_column=target_column,
        normalize=True
    )
    
    print(f"✅ Dataset created with {len(dataset)} samples")
    print(f"✅ Input features: {dataset.num_features}")
    
    # 3. Create data splits
    print("\n🔄 Step 3: Creating train/val/test splits...")
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset, 
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print(f"✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 4. Create data loaders
    print("\n📦 Step 4: Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=5,  # Small batch for the tiny dataset
        num_workers=0
    )
    
    print(f"✅ Data loaders created")
    
    # 5. Create model
    print("\n🧠 Step 5: Creating MLP model...")
    model = create_model(
        model_type='mlp',
        input_dim=dataset.num_features,
        num_classes=data_analysis['num_classes'],
        hidden_dims=[10, 8],  # Small network for small dataset
        dropout=0.1,
        task_type='classification'
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {num_params:,} parameters")
    
    # 6. Train the model
    print("\n🏋️ Step 6: Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,  # Quick training for demo
        optimizer_config={
            'optimizer_type': 'adam',
            'learning_rate': 0.01
        },
        device=device,
        task_type='classification'
    )
    
    # 7. Show results
    print("\n📈 Step 7: Training Results")
    final_train_acc = history['train_accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    print(f"✅ Final Training Accuracy: {final_train_acc:.3f}")
    print(f"✅ Final Validation Accuracy: {final_val_acc:.3f}")
    
    # 8. Make predictions
    print("\n🔮 Step 8: Making predictions...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        inputs, true_labels = sample_batch
        outputs = model(inputs.to(device))
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        true_labels = true_labels.numpy()
    
    print(f"✅ Test predictions: {predictions}")
    print(f"✅ True labels: {true_labels}")
    print(f"✅ Test accuracy: {(predictions == true_labels).mean():.3f}")
    
    print("\n🎉 Example completed successfully!")
    return model, history


def example_regression():
    """Example configuration for a regression task (conceptual)."""
    
    print("\n🔢 Regression Example Configuration:")
    print("=" * 40)
    
    regression_config = {
        "data_type": "tabular",
        "model_type": "mlp",
        "task_type": "regression",
        "target_column": "price",  # Example: predicting house prices
        "normalize": True,
        "hidden_dims": [64, 32],
        "dropout": 0.2,
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 100,
        "optimizer_config": {
            "optimizer_type": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001
        }
    }
    
    print("✅ For regression tasks, change 'task_type' to 'regression'")
    print("✅ The model will output continuous values instead of class probabilities")
    print("✅ Loss function automatically changes to MSE")
    print(f"✅ Example config: {json.dumps(regression_config, indent=2)}")


if __name__ == "__main__":
    # Run the classification example
    model, history = example_classification()
    
    # Show regression configuration
    example_regression()
    
    print(f"\n📚 To use this framework:")
    print(f"   1. Prepare your CSV data with numeric features")
    print(f"   2. Run: python main.py --data_path your_data.csv --config your_config.json")
    print(f"   3. Or use the functions directly as shown in this example") 