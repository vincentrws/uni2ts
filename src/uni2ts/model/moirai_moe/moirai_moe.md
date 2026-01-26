# MOIRAI-MoE Model

## Overview

The MOIRAI-MoE model extends the original MOIRAI architecture with Mixture-of-Experts (MoE) routing for improved efficiency and capacity. It maintains probabilistic forecasting with mixture distributions while adding expertise specialization and autocorrective inference capabilities.

## Components

### Core Module (`module.py`)

**Key Classes:**
- `MoiraiMoEModule`: Core model with MoE transformer encoder

**Key Features:**
- Multi-patch input projections with enhanced feature processing
- Transformer encoder with MoE routing (`use_moe=True`)
- Residual feature projections (`FeatLinear`, residual connections)
- Fed-forward dimension (`d_ff`) for MoE expert capacity
- Causal attention masking for better temporal modeling

**Architecture Enhancements:**
```python
# Enhanced input processing with residual blocks
self.in_proj = MultiInSizeLinear(in_features_ls=patch_sizes, out_features=d_model)
self.res_proj = MultiInSizeLinear(in_features_ls=patch_sizes, out_features=d_model)
self.feat_proj = FeatLinear(in_features_ls=patch_sizes, out_features=d_model)

# Residual input combination
in_reprs = self.in_proj(scaled_target, patch_size)
in_reprs = F.silu(in_reprs)
in_reprs = self.feat_proj(in_reprs, patch_size)
res_reprs = self.res_proj(scaled_target, patch_size)
reprs = in_reprs + res_reprs

# MoE transformer with causal attention
self.encoder = TransformerEncoder(
    d_model,
    num_layers,
    use_moe=True,  # Expert routing
    d_ff=d_ff,     # Feed-forward expert dimension
    var_attn_bias_layer=partial(BinaryAttentionBias),
    time_qk_proj_layer=partial(QueryKeyProjection, proj_layer=RotaryProjection),
)
```

### Forecasting (`forecast.py`)

**Key Classes:**
- `MoiraiMoEForecast`: Inference module with autocorrective generation

**Key Features:**
- Autocorrective forecasting for improved accuracy
- Sequential prediction with feedback loops
- GluonTS integration for production deployment
- Mixture distribution sampling with correction

**Autocorrective Inference:**
- Predicts segments of forecast horizon
- Feeds back predictions as context for next segments
- Corrects prediction errors through iterative refinement
- Causal attention masking ensures temporal coherence

## Architecture Details

### Mixture-of-Experts Routing
- **Purpose**: Conditional computation with expert specialization
- **Mechanism**: Sparse activation of expert subsets per token
- **Benefits**: Increased model capacity without proportional compute increase
- **Implementation**: Integrated into transformer feed-forward layers

### Enhanced Input Processing
- **Residual Projections**: Combines direct and transformed patch features
- **Feature Learning**: Dedicated `FeatLinear` for learning patch representations
- **SiLU Activation**: Smooth non-linear activation for better gradient flow

### Causal Attention Masking
- **Purpose**: Prevent future information leakage during inference
- **Implementation**: `packed_causal_attention_mask(sample_id, time_id)`
- **Benefits**: Enables autoregressive generation patterns

### Autocorrective Forecasting
- **Algorithm**: Predict → Feed back → Correct iteratively
- **Token-by-token**: Predict chunks of future sequence
- **Self-correction**: Use own predictions as improved context
- **Horizon Extension**: Generate arbitrarily long forecasts

## Usage Workflow

### Model Loading
```python
from uni2ts.model.moirai_moe import MoiraiMoEModule

model = MoiraiMoEModule.from_pretrained("Salesforce/moirai-moe-1.0-R-small")
```

### Inference Setup
```python
from uni2ts.model.moirai_moe import MoiraiMoEForecast

forecast = MoiraiMoEForecast(
    prediction_length=96,
    target_dim=1,
    patch_size=32,  # Fixed patch size for MoE
)
predictor = forecast.create_predictor(batch_size=32)
```

### Autocorrective Prediction
```python
# Internal autocorrective flow:
# 1. Initial prediction chunk
# 2. Feed prediction back as context
# 3. Correct and extend prediction
# 4. Repeat until horizon complete

forecasts = predictor.predict(test_dataset)
```

## Key Innovations

### 1. Mixture-of-Experts Integration
- **Conditional Computation**: Experts specialize in different patterns
- **Capacity Scaling**: More parameters with controlled compute
- **Sparse Activation**: 1.5x capacity with <1.2x compute typically

### 2. Enhanced Input Representations
- **Multi-path Projection**: Direct + residual feature learning
- **Feature Fusion**: Combines original and learned representations
- **Non-linear Processing**: SiLU activation for richer representations

### 3. Autocorrective Inference
- **Self-improvement**: Predictions correct themselves iteratively
- **Error Reduction**: Feedback loop reduces accumulated errors
- **Long Horizons**: Enables extended forecasting beyond model limits

### 4. Causal Modeling
- **Temporal Coherence**: Prevents information leakage
- **Generative Capabilities**: Supports autoregressive generation
- **Context Awareness**: Proper handling of temporal relationships

## Performance Characteristics

### Advantages over Base MOIRAI
- **Efficiency**: Better parameter utilization through expert routing
- **Capacity**: Higher effective model capacity
- **Accuracy**: Autocorrective inference improves prediction quality
- **Scalability**: Maintains performance with larger parameter counts

### Computational Considerations
- **Memory**: Slightly higher due to multiple experts
- **Training**: More complex optimization surface
- **Inference**: Autocorrective generation adds latency
- **Deployment**: Requires careful expert routing configuration

## Files Structure

```
src/uni2ts/model/moirai_moe/
├── __init__.py          # Module exports (MoiraiMoEModule, MoiraiMoEForecast)
├── module.py            # Core MoE model implementation
├── forecast.py          # Autocorrective inference and GluonTS integration
```

## Integration Points

### With Core MOIRAI Components
- **Compatible**: Uses same patch processing and attention mechanisms
- **Enhanced**: Adds MoE routing and causal attention
- **Extended**: Autocorrective forecasting capabilities

### With Data Pipeline
- **Same Format**: Compatible with MOIRAI data transformations
- **Causal Masking**: Requires causal attention mask support
- **Sequence Length**: Optimized for patch_size=32 configuration

### With Distribution Module
- **Same Support**: Uses identical mixture distribution outputs
- **Probabilistic**: Maintains uncertainty quantification
- **Sampling**: Compatible with all DistributionOutput classes

## Configuration

### Model Hyperparameters
```python
MoiraiMoEModule(
    distr_output=StudentTOutput(),  # Same as base MOIRAI
    d_model=768,                     # Hidden dimension
    d_ff=3072,                       # Feed-forward with experts
    num_layers=12,                   # Transformer layers
    patch_sizes=[32],               # Typically single patch size
    max_seq_len=512,                # Maximum context
    attn_dropout_p=0.1,            # Attention dropout
    dropout_p=0.1,                 # General dropout
)
```

### MoE-Specific Configuration
- **Expert Count**: Implicit in `d_ff` scaling
- **Routing Strategy**: Sparse token-to-expert mapping
- **Load Balancing**: Automatic expert utilization balancing
- **Training Stability**: Gradient clipping and routing loss

### Inference Configuration
```python
MoiraiMoEForecast(
    prediction_length=96,
    target_dim=1,
    patch_size=32,                  # Fixed for MoE model
    num_samples=100,                # Samples for uncertainty
)
```

## Technical Details

### Expert Routing Mechanism
- **Router**: Learned token-to-expert assignment
- **Top-K**: Activate K experts per token (typically K=2)
- **Load Balance**: Ensures expert utilization
- **Specialization**: Experts learn complementary representations

### Causal Attention Implementation
- **Mask Shape**: `packed_causal_attention_mask(sample_id, time_id)`
- **Temporal Ordering**: Preserves sequence causality
- **Autoregressive**: Enables step-by-step generation
- **Efficiency**: Minimal overhead compared to full attention

### Autocorrective Algorithm
```python
# Pseudocode for autocorrective forecasting
for step in prediction_steps:
    # 1. Predict next segment
    prediction = model.forward(context)
    
    # 2. Concatenate to context
    context = torch.cat([context, prediction], dim=-1)
    
    # 3. Causal masking updates
    attention_mask = update_causal_mask(context.length)
```

## Limitations and Considerations

### Training Challenges
- **Expert Imbalance**: Routing can lead to unused experts
- **Load Balancing**: Requires additional training losses
- **Capacity Utilization**: Not guaranteed to be efficient

### Inference Trade-offs
- **Latency**: Autocorrective loop adds computational overhead
- **Memory**: Stores intermediate predictions
- **Complexity**: More complex than single-shot forecasting

### Compatibility
- **Data Format**: Requires compatible patch processing
- **GluonTS**: Inherits same prediction interface
- **Checkpoint**: Checkpoint format differs from base MOIRAI

This MoE extension provides a pathway to scaling MOIRAI models with improved efficiency and accuracy through specialized expert routing and advanced inference techniques.