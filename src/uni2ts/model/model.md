# Model Module

## Overview

The model module provides complete MOIRAI model implementations, including the original Moirai-1.0-R series, the Mixture-of-Experts variant, and the simplified Moirai-2.0-R. Each variant includes modules for pre-training, fine-tuning, and inference, along with predictor classes for production use.

## MOIRAI-1.0-R Architecture (`moirai/`)

### MoiraiModule (`module.py`)

**Core Components:**
- Multi-patch size input projections
- Transformer encoder with GQA attention
- Mixture distribution output heads
- Support for 5-500 patch sizes

**Key Innovations:**
- Any-variate attention with binary bias
- RoPE position embeddings for time
- Masked encoder pre-training objective
- Zero-shot forecasting capability

### Training Variants

**MoiraiPretrainModule:** For masked sequence modeling
- Handles variable-length input/output pairs
- Implements BERT-style masked prediction
- Supports sequence packing for efficiency

**MoiraiFinetuneModule:** For supervised fine-tuning
- Adapts pre-trained encoder for forecasting
- Adds prediction head and loss computation
- Maintains universal time series capabilities

### Inference (`forecast.py`)

**MoiraiForecast:** Production prediction interface
- `from_pretrained()` loading from HuggingFace Hub
- `create_predictor()` for GPU-optimized inference
- Batch processing with configurable sample counts
- GluonTS compatibility for integration

## MOIRAI-MoE Architecture (`moirai_moe/`)

### Mixture-of-Experts Extension

**Key Features:**
- 32 expert FFN layers with routing
- Expert sharing across layers
- Token-level expert selection (top-2)
- Improved efficiency per parameter

**Benefits:**
- Scales model capacity without proportional compute
- Sparse activation for reduced latency
- Competitive with larger dense models

## MOIRAI-2.0-R Architecture (`moirai2/`)

### Simplified Deterministic Model

**Key Changes:**
- Fixed patch size (16) instead of multiple
- Deterministic quantile regression output
- Reduced complexity for faster inference
- Competitive accuracy with simpler architecture

**Advantages:**
- Lower computational requirements
- Transparent prediction intervals
- Easier deployment and monitoring

## Pre-trained Model Variants

### Size Options
- **Small**: ~14M parameters (Moirai-1.1-R-small)
- **Base**: ~91M parameters (Moirai-1.1-R-base)  
- **Large**: ~311M parameters (Moirai-1.1-R-large)
- **MoE**: Small/Base with expert routing

### Training Datasets
- LOTSA (Large-scale Observation Time Series Archive)
- Diverse domains: weather, economics, traffic, etc.
- Varied frequencies and horizons
- 27B+ time series points

## Model Loading and Usage

### HuggingFace Integration
```python
from uni2ts.model.moirai import MoiraiForecast

# Load pre-trained model
model = MoiraiForecast.from_pretrained("Salesforce/moirai-1.1-R-small")
predictor = model.create_predictor(batch_size=32)

# Zero-shot forecasting
forecasts = predictor.predict(test_data)
```

### Custom Fine-tuning
```python
from uni2ts.model.moirai import MoiraiFinetuneModule, MoiraiModule

# Load and wrap for fine-tuning
model = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")
lightning_module = MoiraiFinetuneModule(model, loss_fn, lr=1e-4)

# Train on custom dataset
trainer.fit(lightning_module, dataloader)
```

## Architecture Components

### Input Processing
- Patch selection based on frequency constraints
- Multi-patch size embeddings
- Normalization and scaling

### Transformer Encoder
- Pre-layer normalization (default)
- Grouped query attention
- Feed-forward with GLU activation
- MoE routing (in MoE variants)

### Output Heads
- Mixture of 4 parametric distributions
- Deterministic quantiles (Moirai-2.0-R)
- Context length up to 512 patches

## Performance Characteristics

### Zero-shot Capabilities
- Competitive accuracy across diverse datasets
- Handles new datasets without fine-tuning
- Robust to distribution shift

### Fine-tuning Benefits
- Improved accuracy on specific domains
- Domain adaptation for specialized forecasting
- Maintains generalization capabilities

### Inference Efficiency
- Batch processing optimizes GPU utilization
- Variable context/prediction lengths
- Production-ready speed and memory usage

## Relationship to Other Modules

### With Module Module
- Uses transformer components and attention mechanisms
- Integrates scalers and embeddings
- Extends with distribution outputs

### With Transform Module
- Compatible with preprocessing pipelines
- Expects patched and normalized inputs
- Prediction masking for training

### With Distribution Module
- Uses mixture distribution outputs for uncertainty
- Supports various parametric families
- Quantile regression for intervals

### With Loss Module
- NLL loss for probabilistic training
- Packed loss functions for efficiency
- Masked prediction support

This model module provides the complete MOIRAI forecasting system with multiple variants optimized for different use cases, from zero-shot universal forecasting to specialized domain fine-tuning.