# MLP Deep Learning Framework for Tabular Data

This project provides a flexible and streamlined framework for training and evaluating Multi-Layer Perceptron (MLP) models on tabular datasets. It handles data analysis, preprocessing, model training, and prediction, focusing exclusively on MLP-based tasks.

## Features

-   **Data Analysis**: Automatically analyzes tabular datasets to extract key information like feature distributions, number of classes, and sample counts.
-   **Data Preprocessing**: Handles numerical feature normalization and target variable encoding for classification/regression.
-   **MLP Model**: Implements a configurable MLP model (`MLPClassifier`) supporting both classification and regression tasks.
-   **Training Pipeline**: Includes standard training loops, optimization, learning rate scheduling, and early stopping.
-   **Evaluation**: Generates predictions and saves them in a submission-friendly format.
-   **Configuration**: Uses JSON files for easy experiment configuration.
-   **Extensible**: Built with modularity in mind (though now focused on MLP).

## Project Structure

```
.
├── main.py                 # Main script to run the framework
├── README.md               # This file
├── example_configs/
│   └── mlp_config.json     # Example configuration for an MLP task
├── sample_data/
│   └── sample_mlp_data.csv # Sample tabular data for testing
├── src/
│   ├── data/
│   │   └── dataset.py      # TabularDataset and data loading utilities
│   ├── models/
│   │   └── architectures.py # MLPClassifier and model creation utilities
│   └── utils/
│       ├── data_utils.py   # Data analysis, splitting, loading utilities
│       └── training.py     # Training loop, prediction, and related utilities
└── outputs/                  # Default directory for saving results (models, logs, predictions)
```

## Prerequisites

-   Python 3.8+
-   PyTorch
-   Pandas
-   NumPy
-   scikit-learn (for StandardScaler and potential future utilities)
-   Matplotlib (for plotting training history)

Install dependencies using:
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## Getting Started

### 1. Prepare Your Data

Your dataset should be a single CSV file.
-   One column must be designated as the target variable.
-   All other columns used for training should be numeric (or will be attempted to be converted/ignored if not specified in `feature_columns`).
-   See `sample_data/sample_mlp_data.csv` for an example. This is a snippet of the Iris dataset.

### 2. Configure Your Experiment

Create a JSON configuration file (or modify `example_configs/mlp_config.json`).

Key configuration options in `mlp_config.json`:

-   `data_type`: Must be `"tabular"`.
-   `model_type`: Must be `"mlp"`.
-   `task_type`: `"classification"` or `"regression"`.
-   `target_column`: Name of the target variable column in your CSV.
-   `feature_columns`: (Optional) List of feature column names. If `null` or omitted, all numeric columns except the `target_column` are used.
-   `normalize`: `true` or `false` to enable/disable feature normalization.
-   `hidden_dims`: List of integers for the sizes of hidden layers in the MLP (e.g., `[128, 64]`).
-   `dropout`: Dropout rate (e.g., `0.3`).
-   `batch_size`, `learning_rate`, `num_epochs`: Standard training parameters.
-   `optimizer_config`, `scheduler_config`, `early_stopping`: Configurations for optimizer, LR scheduler, and early stopping.
-   `input_dim`, `num_classes`: These are typically determined automatically during the analysis/training phase. However, if you are running in `predict_only` mode *without* a preceding training run that saved a configuration, you **must** provide these values in the config based on the model you are loading (e.g., from a `final_run_config.json` of a previous training). `num_classes` is for classification; for regression, the model head defaults to 1 output.

**Example `mlp_config.json` for Iris-like classification:**
```json
{
  "data_type": "tabular",
  "model_type": "mlp",
  "task_type": "classification",
  "target_column": "target",
  "feature_columns": ["feature1", "feature2", "feature3", "feature4"],
  "normalize": true,
  "hidden_dims": [10, 8],
  "dropout": 0.1,
  "batch_size": 5,
  "learning_rate": 0.01,
  "num_epochs": 100,
  "optimizer_config": {
    "optimizer_type": "adam",
    "learning_rate": 0.01
  },
  "scheduler_config": null,
  "early_stopping": {
    "patience": 10,
    "min_delta": 0.001
  },
  "train_ratio": 0.7,
  "val_ratio": 0.15,
  "test_ratio": 0.15,
  "input_dim": null, // Will be auto-filled during training
  "num_classes": null // Will be auto-filled during training
}
```

### 3. Run the Framework

Use `main.py` to run various stages:

**a) Analyze Data Only:**
```bash
python main.py \
    --data_path sample_data/sample_mlp_data.csv \
    --config example_configs/mlp_config.json \
    --output_dir outputs/mlp_analysis_example \
    --analyze_only
```
This will generate a `data_analysis.json` and `data_distribution.png` in the output directory.

**b) Train a Model:**
```bash
python main.py \
    --data_path sample_data/sample_mlp_data.csv \
    --config example_configs/mlp_config.json \
    --output_dir outputs/mlp_training_example
```
This will:
1.  Analyze the data.
2.  Train the MLP model based on the configuration.
3.  Save the best model (`best_model.pth`), training history (`training_history.json`, `training_history.png`), and the final configuration used (`final_run_config.json`) in the output directory.
4.  Generate predictions on the test split and save them (`predictions.csv`).

**c) Predict with a Pre-trained Model:**

First, ensure your configuration file (`example_configs/mlp_config.json` or a copy) has the correct `input_dim` and `num_classes` that match the pre-trained model. You can find these in the `final_run_config.json` from the training output. For the `sample_mlp_data.csv` example, `input_dim` would be 4, and `num_classes` would be 3.

**Example `mlp_config_for_prediction.json` (after training the sample):**
```json
{
  "data_type": "tabular",
  "model_type": "mlp",
  "task_type": "classification",
  "target_column": "target",
  "feature_columns": ["feature1", "feature2", "feature3", "feature4"],
  "normalize": true, // Ensure this matches training
  "hidden_dims": [10, 8], // Must match trained model
  "dropout": 0.1, // Must match trained model
  "batch_size": 5,
  // LR, epochs, optimizer etc. are not used for prediction but config structure is kept
  "input_dim": 4,     // Crucial: Set from trained model's config
  "num_classes": 3    // Crucial: Set from trained model's config (for classification)
}
```

Then run:
```bash
python main.py \
    --data_path path/to/your/new_test_data.csv \
    --config path/to/your/mlp_config_for_prediction.json \
    --output_dir outputs/mlp_prediction_example \
    --model_path outputs/mlp_training_example/best_model.pth \
    --test_path path/to/your/new_test_data.csv \
    --predict_only
```
(Note: For simplicity, `data_path` is still required by the argument parser but not directly used in `predict_only` mode if `test_path` is provided. `test_path` is the actual data source for predictions.) *Correction*: `data_path` in `predict_only` mode is used to prepare the test dataset via `prepare_datasets(args.test_path, config)` if `args.test_path` is given. If you want to use a separate dataset for prediction, point `--test_path` to it. The original `--data_path` argument becomes less relevant in `predict_only` mode if `--test_path` is specified for the actual prediction data. The script currently uses `args.test_path` for loading prediction data.

## Output

The framework generates the following outputs in the specified output directory:
-   `data_analysis.json`: Statistics and analysis of the dataset.
-   `data_distribution.png`: Visualization of data distributions.
-   `final_run_config.json`: The exact configuration used for the run.
-   `best_model.pth`: Saved weights of the best trained model.
-   `training_history.json`: Log of training metrics (loss, accuracy, etc.).
-   `training_history.png`: Plot of training history.
-   `predictions.csv`: Predictions made on the test set.

## Customization

-   **Model Architecture**: Modify `hidden_dims` and `dropout` in the config file. For more advanced changes, edit `MLPClassifier` in `src/models/architectures.py`.
-   **Data Handling**: Adjust `feature_columns`, `target_column`, and `normalize` in the config. For custom preprocessing, extend `TabularDataset` in `src/data/dataset.py`.
-   **Training Process**: Tune hyperparameters like `learning_rate`, `batch_size`, `num_epochs`, and configure `optimizer_config`, `scheduler_config`, `early_stopping` in the config file. Modify `train_model` in `src/utils/training.py` for deeper changes.

This framework is designed to be a starting point for MLP-based deep learning tasks on tabular data.
