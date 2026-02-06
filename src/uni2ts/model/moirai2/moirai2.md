# MOIRAI 2.0 Model

## Overview

The MOIRAI 2.0 model represents a significant simplification and specialization of the original MOIRAI architecture. It abandons probabilistic mixture distributions for deterministic quantile regression, using fixed single patch sizes and simplified input processing. The model focuses on efficient, interpretable forecasting at the cost of uncertainty quantification.

## Components

### Core Module (`module.py`)

**Key Classes:**
- `Moirai2Module`: Simplified deterministic model implementation

**Key Features:**
- Fixed single patch size (no multi-patch complexity)
- Residual block input projections (`ResidualBlock`)
- No distribution output - direct quantile prediction
- Simplified attention with causal masking
- Quantile regression with configurable levels

**Simplified Architecture:**
```python
# Simple residual block projections
self.in_proj = ResidualBlock(
    input_dims=patch_size * 2,  # target + mask
    hidden_dims=d_model,
    output_dims=d_model,
)

# Direct quantile output projection
self.out_proj = ResidualBlock(
    input_dims=d_model,
    hidden_dims=d_model,
    output_dims=num_predict_token * num_quantiles * patch_size,
)

# Training mode returns predictions, eval returns quantiles
if training_mode:
    return preds, scaled_target
else:
    return preds * scale + loc
```

### Forecasting (`forecast.py`)

**Key Classes:**
- `Moirai2Forecast`: Quantile forecasting with recursive generation

**Key Features:**
- Recursive quantile forecasting for long horizons
- Quantile sampling and confidence intervals
- GluonTS integration with `QuantileForecastGenerator`
- Autoregressive extension beyond model prediction windows

**Recursive Forecasting:**
- Initial quantile prediction chunk
- Sample values for next-step prediction input
- Iterate until full horizon is covered
- Quantile computations using torch.quantile

## Architecture Details

### Single Patch Size Design
- **Purpose**: Eliminate complexity of multi-patch selection
- **Benefit**: Faster inference, simpler architecture
- **Trade-off**: Fixed temporal resolution (no frequency adaptation)

### Residual Block Projections
- **Mechanism**: Stack input, SiLU, output with residual connections
- **Efficiency**: Fewer parameters than MultiInSizeLinear
- **Performance**: Comparable expressiveness with single patch

### Direct Quantile Output
- **Method**: Predict quantile values directly (not distribution parameters)
- **Levels**: Configurable quantile levels (e.g., 0.1, 0.5, 0.9)
- **Uncertainty**: Represented through quantile ranges
- **Speed**: No distribution sampling required

### Recursive Extension
- **Algorithm**: Use predicted quantiles as context for further predictions
- **Horizon**: Extend beyond single forward pass limits
- **Quantile Sampling**: Convert quantiles back to point predictions
- **Accumulation**: Build confidence intervals progressively

## Usage Workflow

### Model Initialization
```python
from uni2ts.model.moirai2 import Moirai2Module

model = Moirai2Module(
    d_model=768,
    num_layers=12,
    patch_size=16,
    context_length=512,
    quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
)
```

### Inference Setup
```python
from uni2ts.model.moirai2 import Moirai2Forecast

forecast = Moirai2Forecast(
    prediction_length=96,
    target_dim=1,
    context_length=512,
)

predictor = forecast.create_predictor(batch_size=32)
forecasts = predictor.predict(test_data)
```

### Direct Prediction
```python
# Direct model prediction
predictions = model.predict(past_target, past_observed_target)
# Returns shape: [batch, num_quantiles, prediction_length]
```

## Key Innovations

### 1. Quantile Regression Architecture
- **Deterministic**: Direct quantile prediction vs. distribution parameters
- **Efficient**: One forward pass per prediction
- **Interpretable**: Quantiles provide clear uncertainty bounds

### 2. Simplified Input Processing
- **Fixed Patch**: Single patch size for all frequencies
- **Residual Blocks**: Streamlined projection layers
- **Concatenated Input**: [target, observed_mask] as input channels

### 3. Recursive Forecasting
- **Extension Mechanism**: Predict beyond model capacity
- **Self-Contained**: Uses own predictions for continuation
- **Adaptive Horizon**: Configurable prediction lengths

### 4. Quantile Mathematics
- **Point Estimates**: Median quantile (0.5) as point forecast
- **Confidence Intervals**: Quantile ranges for uncertainty
- **Robust Statistics**: Less sensitive to outliers than distributions

## Performance Characteristics

### Advantages
- **Speed**: Faster inference than probabilistic models
- **Simplicity**: Easier to deploy and understand
- **Memory**: Lower memory footprint
- **Deterministic**: Reproducible predictions

### Limitations
- **Uncertainty**: No full distribution, only quantiles
- **Expressiveness**: Limited to quantile assumptions
- **Flexibility**: Less adaptable to different data characteristics

### Use Cases
- **Production Systems**: Where speed and determinism matter
- **Risk Assessment**: Quantile-based uncertainty bounds
- **Real-time Forecasting**: Low-latency requirements
- **Resource Constraints**: Limited compute environments

## Files Structure

```
src/uni2ts/model/moirai2/
├── __init__.py          # Module exports (Moirai2Module, Moirai2Forecast)
├── module.py            # Core quantile regression model
├── forecast.py          # Recursive forecasting and GluonTS integration
```

## Technical Details

### Quantile Prediction Mechanism
```python
# Training: predict quantiles from context
pred_quantiles = model(context, training=True)  # [batch, seq, num_quantiles*patch]
scaled_pred = pred_quantiles * scale + loc

# Loss computation
quantile_loss = quantile_huber_loss(pred_quantiles, target_quantiles)
```

### Recursive Extension Algorithm
```python
# For horizons longer than model capacity
def extend_forecast(current_context, remaining_steps):
    while remaining_steps > 0:
        # Predict next chunk
        next_quantiles = model.predict_chunk(current_context)
        
        # Sample medians for autoregressive input
        median_predictions = next_quantiles[:, quantile_levels.index(0.5), :]
        
        # Extend context
        current_context = torch.cat([current_context, median_predictions], dim=-1)
        
        remaining_steps -= chunk_size
```

### Input Processing
- **Concatenation**: `[scaled_target, observed_mask.float()]`
- **Padding**: Zero-padding for fixed patch alignment
- **Chunking**: Process sequences in prediction token chunks

### Quantile Loss Function
- **Huber Loss**: Robust quantile regression loss
- **Gradient Scaling**: Account for quantile density
- **Clipping**: Handle quantile boundary conditions

## Configuration

### Model Parameters
```python
Moirai2Module(
    d_model=768,                          # Hidden dimension
    d_ff=3072,                            # Feed-forward dimension
    num_layers=12,                        # Transformer layers
    patch_size=16,                        # Fixed patch size
    max_seq_len=512,                      # Maximum context
    quantile_levels=(0.1, 0.5, 0.9),    # Prediction quantiles
    attn_dropout_p=0.0,                  # No dropout for deterministic
    dropout_p=0.0,                       # No dropout for deterministic
)
```

### Inference Parameters
```python
Moirai2Forecast(
    prediction_length=96,                  # Forecast horizon
    context_length=512,                   # Context window
    target_dim=1,                         # Number of series
    feat_dynamic_real_dim=0,              # Dynamic features
)
```

## Integration Points

### With Existing MOIRAI Ecosystem
- **Different Output**: Quantiles vs. distributions
- **Same Interface**: Compatible transform pipelines
- **GluonTS Support**: Uses QuantileForecastGenerator
- **HuggingFace**: Separate hub checkpoints

### With Data Pipeline
- **Same Transforms**: Compatible with MOIRAI preprocessing
- **Quantile Output**: Different loss functions
- **Fixed Patches**: Simplified patch size handling

### With Evaluation Framework
- **Quantile Metrics**: Uses quantile-specific evaluation
- **CRPS Equivalent**: Quantile-based scoring
- **Benchmarking**: Separate from distribution-based models

## Comparison with Other MOIRAI Variants

| Aspect | MOIRAI 1.0/1.1 | MOIRAI-MoE | MOIRAI 2.0 |
|--------|----------------|------------|------------|
| Output | Probabilistic | Probabilistic | Deterministic |
| Complexity | High | Very High | Low |
| Speed | Medium | Medium | High |
| Uncertainty | Full Distribution | Full Distribution | Quantiles Only |
| Adaptability | High | High | Low |
| Memory | Medium | High | Low |

## Strengths and Trade-offs

### Strengths
1. **Performance**: Optimized for specific use cases
2. **Simplicity**: Easier deployment and maintenance
3. **Speed**: Fast inference without sampling
4. **Resource Efficient**: Lower memory requirements

### Trade-offs
1. **Expressiveness**: Limited to quantile assumptions
2. **Adaptability**: Less flexible than probabilistic models
3. **Uncertainty**: Quantiles provide less information than distributions
4. **Generalization**: May perform worse on diverse datasets

This streamlined version of MOIRAI prioritizes efficiency and interpretability for production forecasting applications where speed and determinism are critical requirements.
