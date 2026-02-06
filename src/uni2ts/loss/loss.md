# Loss Module

## Overview

The loss module provides loss functions designed for efficient training of MOIRAI models with packed sequences. All losses support variable-length sequences, masking, and proper normalization across samples and variates.

## Core Architecture (`packed/_base.py`)

### PackedLoss Base Class
Abstract base class defining the interface for all loss functions supporting packed inputs.

**Key Features:**
- Handles variable-length sequences efficiently
- Supports prediction and observation masking
- Automatic normalization across samples and variates
- Compatible with sequence packing strategy

**Core Method:**
- `__call__()`: Main loss computation with automatic masking and reduction
- `_loss_func()`: Abstract method for specific loss computation
- `reduce_loss()`: Handles loss aggregation across different dimensions

**Masking Support:**
- `prediction_mask`: Denotes which positions are predictions vs history
- `observed_mask`: Handles missing values in targets
- `sample_id`: Groups positions belonging to same time series sample
- `variate_id`: Groups positions belonging to same variate

### Loss Categories

**PackedPointLoss**: For deterministic point forecasts
**PackedQuantileLoss**: For quantile regression outputs  
**PackedDistributionLoss**: For probabilistic distribution outputs

## Loss Functions

### Distribution Losses (`packed/distribution.py`)

**PackedNLLLoss**: Negative Log-Likelihood loss for distributions.

**Key Features:**
- Computes -log P(target | distribution) for each prediction position
- Used for training probabilistic forecast models
- Compatible with all distribution types (Normal, Mixture, etc.)

**Mathematical Formulation:**
```
loss = -pred.log_prob(target)
```

### Point Losses (`packed/point.py`)

**MSELoss, MAELoss, etc.**: Standard regression losses for point predictions.

**Features:**
- Element-wise loss computation
- Support for multivariate predictions
- Automatic masking of unobserved values

### Quantile Losses (`packed/quantile.py`)

**QuantileLoss**: Pinball loss for quantile regression.

**Features:**
- Asymmetric loss function for different quantiles
- Supports multiple quantile levels simultaneously
- Commonly used for forecast uncertainty estimation

### Normalized Losses (`packed/normalized.py`)

**NMSELoss, NMAELoss**: Normalized mean squared/absolute errors.

**Features:**
- Scale-invariant loss functions
- Useful for comparing performance across different scales
- Division by observed value for normalization

### Percentage Losses (`packed/percentage_error.py`)

**MAPELoss**: Mean Absolute Percentage Error loss.

**Features:**
- Percentage-based error metrics
- Robust to scale differences
- Commonly used in forecasting evaluation

## Key Design Patterns

### Packed Sequence Support
All losses are designed for MOIRAI's sequence packing strategy:
- Efficient batching of variable-length sequences
- Shared computation across similar-length sequences
- Automatic masking and weighting

### Multi-Level Reduction
Loss reduction occurs at multiple levels:
1. **Position level**: Loss per prediction position
2. **Sample level**: Average loss per time series sample  
3. **Batch level**: Average loss across all samples

### Mask Handling
Proper handling of different mask types:
- Prediction masks distinguish train/validation regions
- Observation masks handle missing data points
- Sample/variate IDs enable proper normalization

### Type Safety
Extensive use of `jaxtyping` annotations for:
- Tensor shape specifications
- Type checking for different loss categories
- Interface consistency across implementations

## Relationship to Other Modules

### With Distribution Module
- `PackedNLLLoss` directly uses distribution objects
- Supports all parametric and mixture distributions
- Interfaces with `domain_map` and parameter scaling

### With Model Module
- Loss functions used in training loops
- Compatible with different output heads (probabilistic vs deterministic)
- Integrated with PyTorch Lightning training framework

### With Transform Module
- Works with transformed data containing prediction/observation masks
- Compatible with patched sequences and different normalization schemes

### With Evaluation
- Many loss functions correspond to evaluation metrics
- MSE loss relates to MSE metric, NLL loss relates to CRPS
- Can be used for both training and validation

## Implementation Details

### Loss Reduction Logic
The `reduce_loss` method implements complex weighting:
- Counts observed positions per sample
- Normalizes by number of valid predictions
- Handles different variates properly
- Prevents division by zero with safe_div

### Batch Compatibility
Losses are designed to work with:
- Standard batching (all sequences same length)
- Packed batching (variable lengths, shared computation)
- Single sample evaluation
- Distributed training across GPUs

### Memory Efficiency
- In-place operations where possible
- Efficient tensor manipulations using einops
- Minimal overhead during training loops

## Usage Examples

### Standard Training Loss
```python
from uni2ts.loss.packed import PackedNLLLoss
from pytorch_lightning import LightningModule

class MoiraiModule(LightningModule):
    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        target = batch['target']
        loss_fn = PackedNLLLoss()
        loss = loss_fn(
            pred=pred,
            target=target,
            prediction_mask=batch['prediction_mask'],
            observed_mask=batch['observed_mask'],
            sample_id=batch['sample_id'],
            variate_id=batch['variate_id']
        )
        return loss
```

### Custom Loss Implementation
```python
from uni2ts.loss.packed import PackedPointLoss

class CustomMSE(PackedPointLoss):
    def _loss_func(self, pred, target, ...)
        return (pred - target).pow(2)
```

This module provides the loss computation foundation for training accurate and robust time series forecasting models.
