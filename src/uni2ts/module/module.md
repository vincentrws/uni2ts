# Module Directory

The `module` directory contains reusable neural network components and layers that form the backbone of the MOIRAI models. These modules are designed to be flexible, supporting features like sequence packing, Mixture-of-Experts (MoE), and various attention mechanisms.

## Files & Sub-modules

- [`transformer.py`](transformer.py): High-level Transformer encoder implementations.
- [`attention.py`](attention.py): Specialized attention mechanisms (e.g., Grouped Query Attention).
- [`ffn.py`](ffn.py): Feed-forward network variants, including GLU and MoE.
- [`ts_embed.py`](ts_embed.py): Linear layers for handling multiple patch sizes and residual blocks.
- [`packed_scaler.py`](packed_scaler.py): Standardization modules that correctly handle packed sequences.
- [`norm.py`](norm.py): Normalization layers (e.g., RMSNorm).
- [`position/`](position/position.md): Sub-module for temporal and variate position encodings.

## Key Components

### `TransformerEncoder` (in `transformer.py`)

#### Purpose
A highly configurable Transformer encoder that orchestrates multiple `TransformerEncoderLayer`s.

#### Features
- **MoE Support**: Can be configured to use Mixture-of-Experts in the feed-forward layers.
- **Any-variate Attention**: Integrates `BinaryAttentionBias` for handling multiple variates.
- **Rotary Embeddings**: Integrates RoPE for temporal positioning.
- **Flexible Norm**: Supports various normalization layers (LayerNorm, RMSNorm) and pre/post-norm configurations.

### `MultiInSizeLinear` & `MultiOutSizeLinear` (in `ts_embed.py`)

#### Purpose
These modules are central to the "Universal" nature of MOIRAI. They allow the model to accept input patches (or produce output parameters) of different sizes (e.g., 8, 16, 32, 64, 128) using a single logical layer.

#### Logic
Internally, they maintain multiple sets of weights and dynamically select the correct one based on the `patch_size` tensor provided during the forward pass.

### `PackedStdScaler` (in `packed_scaler.py`)

#### Purpose
Standardizes input time series (mean=0, std=1) while correctly ignoring padding and masked prediction tokens across multiple sequences in a packed batch.

### `MoEFeedForward` (in `ffn.py`)

#### Purpose
Implements a sparse Mixture-of-Experts layer where each token is routed to a top-k subset of experts. This allows for significantly increasing the model's parameter count without a proportional increase in FLOPs.

## Inter-dependencies
- **`uni2ts.common.torch_util`**: Used extensively for masking, safe division, and packed attention masks.
- **PyTorch**: All components are standard `nn.Module` subclasses.
- **Einops**: Used for complex tensor rearrangements in attention and scaling.

## Connection Flow
1.  **Embedding**: `MultiInSizeLinear` (or `ResidualBlock`) projects raw time series patches to the hidden dimension.
2.  **Scaling**: `PackedStdScaler` computes local statistics per sequence and normalizes the representations.
3.  **Backbone**: The data passes through `TransformerEncoder`, which calls `GroupedQueryAttention` and `FeedForward`/`MoEFeedForward` in each layer.
4.  **Attention**: Within attention, `QueryKeyProjection` and `AttentionBias` (from the `position` sub-module) are applied.
5.  **Output**: The final hidden state is ready for the task-specific head (e.g., distribution projection).
6.  **Loss/Inference**: The model's predictions are compared with targets using specialized packed losses.
