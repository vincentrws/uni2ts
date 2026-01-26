# Module Module

## Overview

The module module contains the core neural network components used in MOIRAI models. It includes attention mechanisms, feed-forward networks, normalizations, transformers, time series embeddings, and scaling utilities. These components form the building blocks of the universal time series forecasting architecture.

## Core Components

### Transformer Architecture (`transformer.py`)

**TransformerEncoderLayer:** Individual transformer encoder layer with:
- Grouped Query Attention (GQA)
- Feed-forward networks (FFN) with optional GLU or MoE
- Pre/post normalization options
- Configurable attention mechanisms

**TransformerEncoder:** Stacked transformer encoder with:
- Multiple patch size support via grouped query attention
- Variable attention bias layers
- MoE support for efficient scaling
- Time and variate position encodings

### Attention Mechanisms (`attention.py`)

**GroupedQueryAttention:** Advanced attention with:
- Grouped query-key projections for different patch sizes
- Binary attention bias for same/different variates
- RoPE position embeddings for temporal modeling
- Normalization and scaling options

**Key Features:**
- Supports multi-patch size processing
- Efficient GQA for long sequences
- Custom attention biases for domain knowledge

### Feed-Forward Networks (`ffn.py`)

**FeedForward:** Standard transformer FFN
**GatedLinearUnitFeedForward:** GLU activation for efficiency  
**MoEFeedForward:** Mixture of Experts for scaling

**Features:**
- Configurable activation functions
- Expert routing in MoE variants
- Dropout and bias controls

### Time Series Embeddings (`ts_embed.py`)

**MultiInSizeLinear:** Handles multiple patch sizes in projection layers
- Separate weight matrices for different patch sizes
- Efficient dynamic routing based on patch size
- Proper initialization and masking

**MultiOutSizeLinear:** Output projections with size flexibility
- Multiple output sizes for different patch configurations
- Feature size-aware routing

**FeatLinear:** Feature transformation with size awareness

**ResidualBlock:** Skip connection modules for better gradient flow

### Normalization (`norm.py`)

**RMSNorm:** Root mean square normalization
**LayerNorm:** Standard PyTorch layer normalization  
**Custom normalizations** optimized for time series data

### Packed Scalers (`packed_scaler.py`)

**PackedStdScaler:** Standard deviation scaling for packed sequences
**PackedAbsMeanScaler:** Absolute mean scaling
**PackedMinMaxScaler:** Min-max normalization
**GroupedPackedStdScaler:** Group-based scaling (for OHLCV)

**Features:**
- Sequence-aware scaling that respects sample boundaries
- Different scaling strategies for pretrained/fine-tuned models
- Support for collective normalization (OHLCV use case)

### Position Encodings (`position/`)

**AttentionBias:** Custom attention bias implementations
**QueryKeyProjection:** QK projection layers for GQA
**Rotary Position Embedding:** Time-aware position encodings

## Multi-Patch Size Architecture

### Core Innovation
MOIRAI's key feature is supporting multiple patch sizes simultaneously:
- Each patch size gets separate projection layers
- Attention mechanism routes between patch sizes
- Allows single model to handle diverse temporal frequencies

### Implementation Pattern
```python
# Different projection layers for each patch size
self.projections = nn.ModuleDict({
    f'patch_{size}': nn.Linear(patch_input_size, model_dim)
    for size in patch_sizes
})
```

### Attention Bias Computation
- Binary bias distinguishes same vs different variates
- Semantic biases can enhance specific relationships (OHLCV)
- Position encodings capture temporal structure

## Relationship to Other Modules

### With Model Module
- Components are assembled into complete MOIRAI architectures
- Attention and FFN layers form the transformer backbone
- Scalers integrate with input processing

### With Transform Module
- Patch embeddings work with patching transformations
- Scalers used in normalization transformations
- Position encodings complement time index additions

### With Loss Module
- Packed scalers prevent information leakage in packed batches
- Attention mechanisms handle masked predictions

### With Distribution Module
- Output projections connect to distribution heads
- Multi-size linear layers support varied patch outputs

## Design Decisions

### Grouped Query Attention (GQA)
- Reduces attention computation for large models
- Maintains quality with grouped projections
- Optimal for time series with varying sequence lengths

### Mixture of Experts (MoE)
- Scales model capacity efficiently
- Expert routing based on input characteristics
- Reduces computation per token

### Packed Sequences
- Efficient batching of variable-length sequences
- Scalers that respect sequence boundaries
- Prevents information leakage across samples

### Multi-Patch Support
- Single model handles all frequencies
- Separate processing paths for each patch size
- Dynamic routing based on input characteristics

## Usage Patterns

### Standard Transformer Block
```python
from uni2ts.module import TransformerEncoder

encoder = TransformerEncoder(
    d_model=512,
    num_layers=6,
    num_heads=8,
    patch_sizes=(8, 16, 32, 64),
    use_glu=True,
    use_qk_norm=True
)
```

### Scaling Input Data
```python
from uni2ts.module import PackedStdScaler

scaler = PackedStdScaler()
scaled_data = scaler.fit_transform(data, sample_id, variate_id)
```

### Time Series Embeddings
```python
from uni2ts.module import MultiInSizeLinear

embedding = MultiInSizeLinear(
    in_features_ls=(32, 64, 128),  # Different patch sizes
    out_features=512
)
```

This module provides the neural architecture foundation enabling MOIRAI's universal time series forecasting capabilities across diverse domains and scales.