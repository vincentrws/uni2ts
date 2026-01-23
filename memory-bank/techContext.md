# MOIRAI Technical Context

## Technology Stack

### Core Dependencies

#### Deep Learning Framework
- **PyTorch**: 2.1.x - 2.4.x
  - Primary deep learning framework
  - Automatic differentiation, GPU acceleration
  - torch.compile support for optimization

#### Training Framework
- **PyTorch Lightning**: >= 2.0
  - Training loop abstraction
  - Distributed training support
  - Checkpointing and logging
  - Multi-GPU/TPU training

#### Time Series Libraries
- **GluonTS**: ~= 0.14.3
  - Dataset handling and transformations
  - Evaluation metrics and utilities
  - Predictor interface for inference

#### Numerical Computing
- **NumPy**: ~= 1.26.0
  - Array operations and numerical computing
- **SciPy**: ~= 1.11.3
  - Statistical distributions and functions
- **einops**: 0.7.*
  - Tensor manipulation with readable notation

#### Type Safety
- **jaxtyping**: ~= 0.2.24
  - Tensor shape annotations
  - Compile-time type checking
  - Integration with beartype

#### Configuration
- **Hydra**: 1.3
  - Declarative configuration management
  - YAML-based experiment configs
  - Command-line argument override

#### Data Management
- **HuggingFace Datasets**: ~= 2.17.1
  - Large-scale dataset loading
  - LOTSA archive integration
- **HuggingFace Hub**: >= 0.23.0
  - Model weights distribution
  - Pre-trained model loading

#### JAX
- **jax[cpu]**
  - Used for specific operations
  - CPU-only installation to avoid conflicts

### Development Dependencies

#### Testing
- **pytest**: 7.4.3
  - Unit and integration tests
- **pytest_timeout**: 2.2.0
  - Test timeout management

#### Code Quality
- **black[jupyter]**: 24.2.0
  - Code formatting
  - Jupyter notebook support
- **isort**
  - Import sorting
- **pre-commit**
  - Git hooks for code quality
  - Automated checks before commits

#### Build System
- **hatch**
  - Build tool for Python packages
  - Version management

### Optional Dependencies

#### Notebooks
- **jupyter**
  - Interactive notebook environment
- **ipywidgets**
  - Interactive UI components
- **matplotlib**
  - Plotting and visualization

#### LOTSA Data Building
- **buildings_bench**
  - Buildings dataset support
- **pyreadr**
  - R data file reading
- **tables**
  - HDF5 data format support
- **subseasonal-data**
  - Subseasonal forecasting data

## Development Setup

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts
```

#### 2. Create Virtual Environment
```bash
virtualenv venv
source venv/bin/activate  # CRITICAL: Must activate before any commands
```

#### 3. Install from Source
```bash
pip install -e '.[notebook]'
```

#### 4. Create Environment File
```bash
touch .env
```

### Environment Variables

#### Required Variables
```bash
# Custom data path for processed datasets
CUSTOM_DATA_PATH=/path/to/custom/data

# LOTSA dataset path for pre-training
LOTSA_V1_PATH=/path/to/lotsa/data

# Long Sequence Forecasting benchmark path
LSF_PATH=/path/to/tslib/dataset
```

#### Setting Variables
```bash
echo "CUSTOM_DATA_PATH=/path/to/save" >> .env
echo "LOTSA_V1_PATH=/path/to/lotsa" >> .env
```

### Project Structure

```
uni2ts/
├── src/uni2ts/           # Source code
│   ├── model/            # Model implementations
│   ├── data/             # Data loading
│   ├── module/           # Neural network components
│   ├── distribution/     # Probabilistic distributions
│   ├── loss/             # Loss functions
│   ├── optim/            # Optimizers
│   ├── transform/        # Data transformations
│   └── eval_util/        # Evaluation utilities
├── cli/                  # Command-line interface
│   ├── train.py          # Training script
│   ├── eval.py           # Evaluation script
│   └── conf/             # Hydra configurations
│       ├── pretrain/     # Pre-training configs
│       ├── finetune/     # Fine-tuning configs
│       └── eval/         # Evaluation configs
├── test/                 # Test suite
├── example/              # Jupyter notebooks
├── docs/                 # Documentation
└── project/              # Specific projects
    ├── moirai-1/         # Moirai-1.0 experiments
    ├── moirai-moe-1/     # Moirai-MoE experiments
    └── benchmarks/       # Benchmark scripts
```

## Technical Constraints

### Python Version
- **Minimum**: Python 3.10
- **Recommended**: Python 3.10 or 3.11
- **Reason**: Modern type hints, f-string features

### PyTorch Version
- **Range**: 2.1.x to 2.4.x
- **Constraint**: Upper bound for compatibility
- **Reason**: API stability, breaking changes in 2.5+

### Hardware Requirements

#### Training
- **GPU**: NVIDIA GPU with CUDA support
- **Memory**: 16GB+ for base model, 32GB+ for large model
- **Storage**: 100GB+ for LOTSA dataset
- **Compute**: Multi-GPU recommended for pre-training

#### Inference
- **GPU**: Optional but recommended for speed
- **Memory**: 8GB+ sufficient for inference
- **CPU**: Acceptable for small batches

### Data Constraints

#### Sequence Length
- **Maximum**: 512 tokens (patches)
- **Context**: Variable, typically 200-1000 patches
- **Horizon**: Variable, typically 20-200 patches

#### Patch Sizes
- **Supported**: 8, 16, 32, 64, 128
- **Selection**: Based on frequency
- **Auto-selection**: Available for Moirai-1.0-R

#### Batch Size
- **Training**: 32-128 depending on GPU memory
- **Inference**: 32-256 for efficiency
- **Packing**: Affects effective batch size

## Tool Usage Patterns

### Command-Line Interface

#### Training
```bash
# Pre-training
python -m cli.train \
  -cp conf/pretrain \
  run_name=first_run \
  model=moirai_small \
  data=lotsa_v1_unweighted

# Fine-tuning
python -m cli.train \
  -cp conf/finetune \
  exp_name=example_lsf \
  run_name=example_run \
  model=moirai_1.0_R_small \
  model.patch_size=32 \
  model.context_length=1000 \
  model.prediction_length=96 \
  data=etth1 \
  data.mode=S
```

#### Evaluation
```bash
# Evaluate custom dataset
python -m cli.eval \
  run_name=example_eval_1 \
  model=moirai_1.0_R_small \
  model.patch_size=32 \
  model.context_length=1000 \
  data=etth1_test

# Evaluate benchmark dataset
python -m cli.eval \
  run_name=example_eval_2 \
  model=moirai_1.0_R_small \
  model.patch_size=32 \
  model.context_length=1000 \
  data=lsf_test \
  data.dataset_name=ETTh1 \
  data.prediction_length=96
```

### Data Processing

#### Simple Dataset Builder
```bash
# Process wide format dataset
python -m uni2ts.data.builder.simple ETTh1 dataset/ETT-small/ETTh1.csv --dataset_type wide

# With train/validation split
python -m uni2ts.data.builder.simple ETTh1 dataset/ETT-small/ETTh1.csv \
  --date_offset '2017-10-23 23:00:00'

# With normalization
python -m uni2ts.data.builder.simple ETTh1 dataset/ETT-small/ETTh1.csv \
  --date_offset '2017-10-23 23:00:00' \
  --normalize
```

### Python API Usage

#### Zero-Shot Forecasting
```python
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# Load model
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
    prediction_length=96,
    context_length=512,
    patch_size=32,
    num_samples=100,
    target_dim=1,
)

# Create predictor
predictor = model.create_predictor(batch_size=32)

# Load and split data
ds = PandasDataset(dict(df))
train, test_template = split(ds, offset=-100)
test_data = test_template.generate_instances(
    prediction_length=96,
    windows=10,
    distance=96
)

# Predict
forecasts = predictor.predict(test_data.input)
```

#### Fine-Tuning
```python
from pytorch_lightning import Trainer
from uni2ts.model.moirai.finetune import MoiraiFinetuneModule

# Load pre-trained model
model = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")

# Create Lightning module
lit_model = MoiraiFinetuneModule(
    model=model,
    loss_fn=PackedNLLLoss(),
    lr=1e-4,
)

# Train
trainer = Trainer(max_epochs=10, accelerator="gpu")
trainer.fit(lit_model, train_dataloader, val_dataloader)
```

### Configuration Management

#### Hydra Config Structure
```yaml
# cli/conf/finetune/default.yaml
defaults:
  - model: moirai_1.0_R_small
  - data: etth1
  - val_data: etth1

exp_name: default
run_name: ${now:%Y-%m-%d-%H-%M-%S}

trainer:
  max_epochs: 10
  accelerator: auto
  precision: 16
```

#### Overriding Configs
```bash
# Override individual parameters
python -m cli.train \
  model.patch_size=64 \
  model.context_length=2048 \
  trainer.max_epochs=20

# Override entire config group
python -m cli.train \
  model=moirai_1.1_R_base \
  data=custom_dataset
```

## Performance Optimization

### Training Optimizations

#### Sequence Packing
- Reduces padding waste from ~61% to <0.4%
- Enables larger effective batch sizes
- Implemented in `PackCollate` class

#### Mixed Precision
- Enabled via Lightning precision=16
- Reduces memory usage by ~50%
- Minimal impact on model quality

#### Gradient Checkpointing
- Trade compute for memory
- Enables training of larger models
- Configurable in Lightning module

### Inference Optimizations

#### Batch Processing
- Default batch size: 32
- Larger batches improve throughput
- Memory-bound vs. compute-bound trade-off

#### Flash Attention
- Optimized attention computation
- Available in PyTorch 2.0+
- Automatically used when available

#### Model Compilation
- PyTorch 2.0 torch.compile
- 10-30% speedup for inference
- May have longer first run (compilation)

## Testing Strategy

### Test Structure
```
test/
├── conftest.py           # Pytest fixtures
├── fixture/              # Test fixtures
│   └── fixture.py        # Data and model fixtures
├── common/               # Common utilities tests
├── data/                 # Data loading tests
├── distribution/         # Distribution tests
├── loss/                 # Loss function tests
├── model/                # Model tests
├── module/               # Neural network module tests
└── transform/            # Transformation tests
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest test/model/test_moirai.py

# Run with coverage
pytest --cov=src/uni2ts

# Run with timeout
pytest --timeout=60
```

## Deployment Considerations

### Model Serving
- Use `MoiraiForecast.create_predictor()` for production
- Batch inference for efficiency
- GPU recommended for latency-sensitive applications

### Containerization
- Base image: Python 3.10+ with CUDA support
- Install via pip: `pip install uni2ts`
- Download model weights from HuggingFace Hub

### Monitoring
- Track prediction latency
- Monitor GPU memory usage
- Log prediction distributions for analysis

## Troubleshooting

### Common Issues

#### Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use smaller model variant

#### Slow Training
- Enable mixed precision
- Use sequence packing
- Increase number of workers

#### Import Errors
- Ensure venv is activated
- Verify installation with `pip show uni2ts`
- Check Python version compatibility

#### CUDA Errors
- Verify CUDA installation
- Check PyTorch CUDA support: `torch.cuda.is_available()`
- Update GPU drivers if needed