# MOIRAI2 Model Module

The `moirai2` module implements the second generation of the MOIRAI architecture. It features a simplified design focusing on quantile regression and deterministic forecasting, while retaining the core universal capabilities of the foundation model.

## Files

- [`module.py`](module.py): Defines the `Moirai2Module`.
- [`forecast.py`](forecast.py): Provides the forecasting wrapper for MOIRAI2.

## Key Classes

### `Moirai2Module` (in `module.py`)

A simplified Transformer-based model optimized for quantile forecasting.

#### Architecture Key Features
- **Fixed Patch Size**: Unlike MOIRAI 1.0, MOIRAI2 typically uses a fixed patch size (e.g., 16 or 32).
- **Residual Projections**: Uses `ResidualBlock` from `uni2ts.module.ts_embed` for both input and output projections, replacing the simpler linear layers of the first version.
- **Input Composition**: Concatenates scaled targets and observed masks directly as input features before projection.
- **Causal Attention**: Uses `packed_causal_attention_mask` to ensure tokens only attend to the past during decoding.
- **Quantile Output**: The output projection layer maps hidden states directly to multiple quantile levels across the patch dimension.

### `Moirai2Forecast` (in `forecast.py`)

#### Purpose
Orchestrates inference for MOIRAI2 models. Since MOIRAI2 uses quantile regression, the forecasting process is deterministic (returning quantiles) rather than sampling-based, though it still integrates with GluonTS.

## Inter-dependencies
- **`uni2ts.module.ts_embed`**: Uses the `ResidualBlock` for high-capacity projections.
- **`uni2ts.module.transformer`**: Uses the standard `TransformerEncoder`.
- **`uni2ts.common.torch_util`**: Uses `packed_causal_attention_mask` and `PackedStdScaler`.

## Connection Flow
1.  **Normalization**: `PackedStdScaler` standardizes the input time series.
2.  **Input Preparation**: Targets and masks are concatenated.
3.  **Embedding**: `ResidualBlock` projects the combined features into the transformer's hidden space.
4.  **Encoding**: Transformer layers process the sequence with causal masking.
5.  **Quantile Mapping**: The final `ResidualBlock` projects hidden states to the `(num_quantiles * patch_size)` dimension.
6.  **De-normalization**: Predicted values are scaled and shifted back to the original data range.
7.  **Output**: Returns deterministic quantile forecasts.
