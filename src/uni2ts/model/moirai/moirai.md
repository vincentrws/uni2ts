# MOIRAI Model (Moirai 1.0/1.1-R)

## Overview

The MOIRAI model implements the original time series foundation model architecture from Salesforce Research. It's a probabilistic forecasting model that uses multi-patch projection, transformer encoder with binary attention bias, and mixture distribution output. The model supports both pre-training and fine-tuning workflows.

## Components

### Core Module (`module.py`)

**Key Classes:**
- `MoiraiModule`: Main model implementation inheriting from PyTorch Lightning Module and HuggingFace Hub mixin

**Key Features:**
- Multi-patch size input projections (`MultiInSizeLinear`)
- Transformer encoder with binary attention bias and rotary position embeddings
- Mixture distribution output (`DistributionOutput`)
- Scaling with `PackedStdScaler`

**Architecture:**
```python
# Input projection layers for different patch sizes
self.in_proj = MultiInSizeLinear(in_features_ls=patch_sizes, out_features=d_model)

# Transformer encoder with attention mechanisms
self.encoder = TransformerEncoder(
    d_model,
    num_layers,
    var_attn_bias_layer=partial(BinaryAttentionBias),
    time_qk_proj_layer=partial(QueryKeyProjection, proj_layer=RotaryProjection),
)

# Distribution output head
self.distr_output = distr_output
self.param_proj = self.distr_output.get_param_proj(d_model, patch_sizes)
```

### Pre-Training (`pretrain.py`)

**Key Classes:**
- `MoiraiPretrain`: PyTorch Lightning module for self-supervised pre-training

**Functionality:**
- Masked prediction training objective
- Dynamic masking ratios and patch constraints
- Optimizer with weight decay parameter groups
- Comprehensive data transformation pipelines

**Training Process:**
- Random masking of input sequence segments
- Predict masked portions using transformer decoder
- Negative log-likelihood loss on mixture distribution
- Cosine annealing learning rate schedule

### Fine-Tuning (`finetune.py`)

**Key Classes:**
- `MoiraiFinetune`: PyTorch Lightning module for supervised fine-tuning
- `SetTrainableParamsMixin`: Parameter freezing utilities

**Functionality:**
- Configurable fine-tuning patterns (full, freeze_ffn, head_only)
- Flexible optimizer group configuration
- Comprehensive transform chains for train/val splits
- Support for masked prediction with controlled horizons

**Fine-Tuning Patterns:**
- **full**: Train all parameters (default)
- **freeze_ffn**: Freeze feed-forward network layers
- **head_only**: Only train distribution output parameters

### Forecasting (`forecast.py`)

**Key Classes:**
- `MoiraiForecast`: PyTorch Lightning module for inference and evaluation

**Functionality:**
- GluonTS integration for production forecasting
- Auto patch size selection during inference
- Probabilistic sampling from mixture distributions
- Rolling window evaluation support

**Inference Features:**
- Automatic patch size optimization
- Batch prediction with uncertainty quantification
- Data conversion from GluonTS format to internal tensors
- Efficient prediction token generation

## Architecture Details

### Multi-Patch Projection
- **Purpose**: Handle different temporal frequencies in single model
- **Mechanism**: Multiple linear projections for patch sizes (8, 16, 32, 64, 128)
- **Selection**: Based on input sequence length and frequency

### Attention Mechanisms
- **Binary Attention Bias**: Same/different variate distinction
- **Rotary Position Embeddings**: Time-aware position encoding
- **Causal Relationships**: Encodes temporal dependencies

### Distribution Output
- **Mixture Model**: Four parametric distributions (Student-t, Negative Binomial, Log-normal, Normal)
- **Automatic Selection**: Distribution choice based on data characteristics
- **Uncertainty Quantification**: Provides confidence intervals for forecasts

### Training Objectives
- **Masked Encoder**: BERT-style masked prediction pre-training
- **NLL Loss**: Negative log-likelihood for probabilistic distributions
- **Sequence Packing**: Efficient batching of variable-length sequences

## Usage Workflow

### Pre-Training
```python
# Configure pre-training
pretrain_module = MoiraiPretrain(
    min_patches=10,
    min_mask_ratio=0.1,
    max_mask_ratio=0.5,
    module_kwargs={"distr_output": StudentTOutput()},
)

# Train on LOTSA dataset
trainer.fit(pretrain_module, train_dataloader)
```

### Fine-Tuning
```python
# Configure fine-tuning
finetune_module = MoiraiFinetune(
    min_patches=5,
    min_mask_ratio=0.2,
    max_mask_ratio=0.4,
    module_kwargs={"distr_output": StudentTOutput()},
    finetune_pattern="full",
)

# Fine-tune on custom dataset
trainer.fit(finetune_module, train_dataloader, val_dataloader)
```

### Inference
```python
# Create predictor
forecast = MoiraiForecast(
    prediction_length=96,
    target_dim=1,
    module_kwargs={"distr_output": StudentTOutput()},
)
predictor = forecast.create_predictor(batch_size=32)

# Generate forecasts
forecasts = predictor.predict(test_data)
```

## Key Innovations

### 1. Multi-Patch Size Projection
- Handles unlimited patch sizes in single model
- Automatic patch size selection for different frequencies
- Optimizes compute vs. temporal detail trade-off

### 2. Binary Attention Bias
- Learns relationships between different variates
- Equivariant to variable ordering
- Scales to high-dimensional multivariate data

### 3. Mixture Distribution Output
- Universal coverage of different data distributions
- Automatic distribution selection
- Superior uncertainty quantification

### 4. Sequence Packing
- Reduces Padding waste from 61% to <0.4%
- Enables large-batch training
- Dramatically improves training efficiency

## Data Flow

### Training:
`Raw Dataset → Builder → Indexer → Transforms → Packed Tensors → Model → Loss → Optimizer Update`

### Inference:
`GluonTS Dataset → Predictor → Data Conversion → Patches → Model → Distributions → Samples → Forecasts`

## Configuration

### Model Hyperparameters
```python
MoiraiModule(
    distr_output=StudentTOutput(),  # Distribution family
    d_model=768,                     # Model hidden dimension
    num_layers=12,                   # Transformer layers
    patch_sizes=[32, 64, 128],      # Supported patch sizes
    max_seq_len=512,                # Maximum sequence length
    attn_dropout_p=0.1,             # Attention dropout
    dropout_p=0.1,                  # General dropout
)
```

### Training Configuration
- **Pre-training**: 10-50 epochs on LOTSA datasets
- **Fine-tuning**: 1-10 epochs on target datasets
- **Learning Rate**: 1e-3 base with warmup and cosine decay
- **Batch Size**: 32-512 depending on GPU memory

## Performance Characteristics

### Strengths
- **Universal**: Works across all time series domains
- **Efficient**: Multi-task learning reduces per-task training
- **Scalable**: Handles sequences up to 512 tokens
- **Probabilistic**: Provides uncertainty estimates

### Limitations
- **Memory**: Large models require substantial GPU memory
- **Specialization**: May benefit from domain-specific fine-tuning
- **Complexity**: Setup requires understanding of patching and masking

## Files Structure

```
src/uni2ts/model/moirai/
├── __init__.py          # Module exports
├── module.py            # Core MoiraiModule implementation
├── pretrain.py          # Self-supervised pre-training
├── finetune.py          # Supervised fine-tuning
├── forecast.py          # Inference and GluonTS integration
```

## Integration Points

### With Builder Module
- Uses `SimpleDatasetBuilder` and `LOTSADatasetBuilder`
- Requires HuggingFace-overlapping dataset format

### With Data Module
- Integrates with `FinetuneDataset`, `PretrainDataset`
- Uses `HuggingFaceDatasetIndexer` for data loading

### With Transform Module
- Extensive use of transformation pipelines
- Custom transform chains for each training stage

### With Distribution Module
- Supports all `DistributionOutput` implementations
- Mixture distributions for multi-domain support

This architecture forms the foundation of the MOIRAI time series forecasting system, enabling universal forecasting capabilities through innovative attention mechanisms and probabilistic modeling.