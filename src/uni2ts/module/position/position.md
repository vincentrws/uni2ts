# Position Module

The `module.position` sub-module implements various position encoding and attention bias mechanisms. These components are used within the Transformer backbone to provide temporal and variate information to the model.

## Files

- [`attn_bias.py`](attn_bias.py): Implements attention bias layers (e.g., Binary, Relative).
- [`attn_projection.py`](attn_projection.py): Implements query/key projections with position encodings (e.g., RoPE).
- [`additive.py`](additive.py): (Not read, but likely implements standard additive position embeddings).

## Key Classes

### `AttentionBias` (in `attn_bias.py`)

An abstract base class for adding biases to the attention score matrix.

#### `BinaryAttentionBias`
- **Purpose**: Used to distinguish between tokens belonging to the same variate vs. different variates.
- **Logic**: It learns two bias values (one for same-id, one for different-id) and adds them to the attention logits. This is crucial for "Any-variate" attention.

#### `LinearAttentionBias`
- **Purpose**: Implements a simple relative position bias that decays linearly with distance.

### `Projection` & `QueryKeyProjection` (in `attn_projection.py`)

#### `RotaryProjection` (RoPE)
- **Purpose**: Implements Rotary Position Embeddings.
- **Logic**: It rotates the query and key representations based on their temporal index (`time_id`). This allows the model to capture relative temporal relationships effectively across long sequences.

#### `LearnedProjection`
- **Purpose**: A learnable position-wise linear projection.

#### `QueryKeyProjection`
- **Purpose**: A wrapper that applies a specific `Projection` (like RoPE) to queries and keys.
- **Partial Factor**: It supports applying the projection to only a fraction of the hidden dimension, which can sometimes improve stability or performance.

## Inter-dependencies
- **`uni2ts.module.attention`**: These components are instantiated and called within the attention layers.
- **PyTorch**: Uses standard `nn.Module` and buffer/parameter management.
- **Jaxtyping**: Annotates complex 5D/6D tensors used in multi-head/grouped attention.

## Connection Flow
1.  **Encoder Layer**: During the multi-head attention step, hidden representations are projected to queries and keys.
2.  **Temporal Encoding**: `QueryKeyProjection` (typically with `RotaryProjection`) is applied to these queries and keys using the `time_id`.
3.  **Variate Awareness**: The `BinaryAttentionBias` is computed using the `variate_id` (and `sample_id` for packed sequences).
4.  **Score Computation**: The attention scores are computed as `(Q @ K) + bias`, combining temporal position and variate relationship information.
