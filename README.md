# Flexible Deep Learning Quiz Framework

A comprehensive, modular PyTorch-based framework for machine learning quizzes that automatically adapts to different data types (images, text, tabular) and provides optimal model configurations.

## 🚀 Features

- **Auto-Detection**: Automatically detects data type and provides optimal configurations
- **Multi-Modal Support**: Handles images, text, and tabular data seamlessly
- **Transfer Learning**: Built-in support for pre-trained models (ResNet, BERT, etc.)
- **Functional Programming**: Modular, reusable components following functional principles
- **Easy Integration**: Simple API for both CLI and Jupyter notebook usage
- **Production Ready**: Includes early stopping, learning rate scheduling, and model checkpointing

## 📁 Project Structure

```
quiz/
├── requirements.txt          # Dependencies
├── main.py                  # CLI entry point
├── README.md               # This file
├── example_notebook.ipynb  # Interactive example
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py      # Flexible dataset implementations
│   ├── models/
│   │   ├── __init__.py
│   │   └── architectures.py # Model architectures
│   └── utils/
│       ├── __init__.py
│       ├── training.py     # Training utilities
│       └── data_utils.py   # Data processing utilities
└── outputs/               # Generated outputs (created automatically)
```

## 🛠️ Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download NLTK data (for text processing):**
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## 📊 Supported Data Types

### 1. Image Data
- **Folder Structure**: Organized by class folders or CSV with image paths
- **Formats**: JPG, PNG, BMP, TIFF
- **Models**: ResNet, EfficientNet, Vision Transformer
- **Features**: Auto data augmentation, transfer learning

### 2. Text Data
- **Format**: CSV with text and label columns
- **Models**: BERT, RoBERTa, DistilBERT
- **Features**: Automatic tokenization, sequence padding

### 3. Tabular Data
- **Format**: CSV with numerical/categorical features
- **Models**: Multi-layer Perceptron (MLP)
- **Features**: Auto preprocessing, scaling, encoding

## 🎯 Quick Start

### Method 1: Command Line Interface

#### 1. Analyze Your Data
```bash
python main.py --data_path ./your_dataset --analyze_only
```

#### 2. Train Model
```bash
python main.py --data_path ./your_dataset --output_dir ./results
```

#### 3. Predict Only
```bash
python main.py --predict_only --model_path ./results/best_model.pth --test_path ./test_data --config ./results/final_config.json
```

### Method 2: Jupyter Notebook

1. Open `example_notebook.ipynb`
2. Change `DATA_PATH` to your dataset path
3. Run all cells

### Method 3: Python Script

```python
from src.utils.data_utils import load_and_analyze_data, get_recommended_config
from src.data.dataset import create_dataset
from src.models.architectures import create_model
from src.utils.training import train_model

# Analyze data
data_analysis = load_and_analyze_data("./your_dataset")
config = get_recommended_config(data_analysis)

# Create dataset and model
dataset = create_dataset(data_type=config['data_type'], data_path="./your_dataset")
model = create_model(model_type=config['model_type'], num_classes=config['num_classes'])

# Train (see example_notebook.ipynb for complete example)
```

## 📋 Configuration Options

The framework automatically generates optimal configurations, but you can customize:

### Image Classification
```json
{
  "data_type": "image",
  "model_type": "image_classifier",
  "backbone": "resnet18",
  "batch_size": 32,
  "learning_rate": 1e-3,
  "num_epochs": 50,
  "early_stopping": {"patience": 10}
}
```

### Text Classification
```json
{
  "data_type": "text",
  "model_type": "text_classifier",
  "model_name": "bert-base-uncased",
  "max_length": 256,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "num_epochs": 10
}
```

### Tabular Data
```json
{
  "data_type": "tabular",
  "model_type": "mlp",
  "hidden_dims": [256, 128],
  "batch_size": 64,
  "learning_rate": 1e-3,
  "num_epochs": 100
}
```

## 📈 Expected Outputs

After running the framework, you'll get:

```
outputs/
├── data_analysis.json      # Dataset statistics and insights
├── final_config.json       # Configuration used for training
├── best_model.pth         # Trained model weights
├── training_history.json   # Training metrics per epoch
├── training_history.png    # Training plots
├── data_distribution.png   # Data visualization
└── predictions.csv         # Test set predictions
```

## 🎲 Usage Examples

### Example 1: CIFAR-10 Style Dataset
```bash
# Dataset structure:
# dataset/
# ├── train/
# │   ├── class1/
# │   │   ├── img1.jpg
# │   │   └── img2.jpg
# │   └── class2/
# │       ├── img3.jpg
# │       └── img4.jpg

python main.py --data_path ./dataset/train
```

### Example 2: Text Classification
```bash
# CSV format:
# text,label
# "This is great!",positive
# "This is bad!",negative

python main.py --data_path ./sentiment_data.csv
```

### Example 3: Tabular Classification
```bash
# CSV format:
# feature1,feature2,feature3,target
# 1.2,3.4,5.6,0
# 2.3,4.5,6.7,1

python main.py --data_path ./tabular_data.csv
```

### Example 4: Custom Configuration
```bash
# Create custom_config.json with your settings
python main.py --data_path ./dataset --config ./custom_config.json
```

## 🔧 Advanced Features

### 1. Model Ensemble
```python
# Train multiple models with different configurations
models = []
for backbone in ['resnet18', 'resnet50', 'efficientnet_b0']:
    config['backbone'] = backbone
    model = create_model_from_config(config, num_classes)
    # Train model...
    models.append(model)

# Ensemble predictions
ensemble_predictions = torch.stack([model(x) for model in models]).mean(0)
```

### 2. Custom Data Preprocessing
```python
from src.data.dataset import FlexibleImageDataset
from torchvision import transforms

# Custom transform pipeline
custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = FlexibleImageDataset(data_path="./data", transform=custom_transform)
```

### 3. Custom Model Architecture
```python
from src.models.architectures import FlexibleImageClassifier

# Custom classifier head
custom_model = FlexibleImageClassifier(
    num_classes=10,
    backbone='resnet50',
    classifier_hidden_dims=[512, 256, 128]
)
```

## 🎯 Quiz-Specific Tips

1. **Unknown Data Type**: The framework will analyze and recommend optimal settings
2. **Limited Time**: Use `--analyze_only` first to understand your data quickly
3. **Memory Constraints**: Reduce `batch_size` in configuration
4. **Quick Testing**: Set `num_epochs` to 5-10 for rapid prototyping
5. **Best Performance**: Use recommended configurations but enable early stopping

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller model
```python
config['batch_size'] = 16  # Reduce from 32
config['backbone'] = 'resnet18'  # Instead of resnet50
```

2. **Slow Training**: Enable mixed precision or reduce model size
```python
# Add to training config
config['mixed_precision'] = True
```

3. **Poor Performance**: Check data quality and try different architectures
```python
# For imbalanced datasets
config['class_weights'] = True
```

## 📝 Submission Formats

The framework supports multiple submission formats:

- **Simple**: Plain text file with predictions
- **Kaggle**: CSV with id and prediction columns
- **Indexed**: CSV with custom column names

```bash
python main.py --data_path ./test_data --predict_only --submission_format kaggle
```

## 🤝 Contributing

The framework follows functional programming principles:

1. **Pure Functions**: Functions should be side-effect free when possible
2. **Modularity**: Each module has a single responsibility
3. **Immutability**: Avoid modifying input parameters
4. **Composability**: Functions should be easily combinable

## 📚 References

- PyTorch Documentation: https://pytorch.org/docs/
- Transformers Library: https://huggingface.co/transformers/
- Transfer Learning Guide: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## 🏆 Performance Tips

1. **Use Transfer Learning**: Pre-trained models often perform better
2. **Data Augmentation**: Automatically applied for image data
3. **Early Stopping**: Prevents overfitting and saves time
4. **Proper Validation**: Framework automatically creates validation splits
5. **Hyperparameter Tuning**: Start with recommended configs, then fine-tune

---

**Happy Learning!** 🎓

For questions or issues, check the example notebook or analyze your data first using `--analyze_only` flag. 