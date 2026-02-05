# MOIRAI-MoE Model Module

The `moirai_moe` module implements the Mixture-of-Experts (MoE) variant of the MOIRAI foundation model. It leverages sparse expert routing within the Transformer blocks to increase model capacity while maintaining efficient compute.

## Files

- [`module.py`](module.py): Defines the `MoiraiMoEModule`.
- [`forecast.py`](forecast.py): Provides the forecasting wrapper for the MoE variant.

## Key Classes

### `MoiraiMoEModule` (in `module.py`)

A Transformer-based model that incorporates Mixture-of-Experts layers.

#### Architecture Differences from Base MOIRAI
- **MoE Backbone**: The `TransformerEncoder` is configured with `use_moe=True`, enabling sparse routing in the feed-forward layers.
- **Enhanced Input Projection**: Uses a combination of `MultiInSizeLinear` and `FeatLinear` for projecting input patches.
- **Residual Projection**: Includes a `res_proj` layer (another `MultiInSizeLinear`) for managing residual connections from the input to the encoder.
- **Activation**: Uses SiLU activation in the projection stage.

### `MoiraiMoEForecast` (in `forecast.py`)

#### Purpose
Similar to `MoiraiForecast`, it provides a GluonTS-compatible interface for the MoE model, supporting probabilistic forecasting and auto-patch selection.

## Inter-dependencies
- **`uni2ts.module.transformer`**: Relies on the MoE-enabled `TransformerEncoder`.
- **`uni2ts.module.ts_embed`**: Uses `FeatLinear` and `MultiInSizeLinear`.
- **`uni2ts.common.torch_util`**: Uses `packed_causal_attention_mask` for inference.
- **`uni2ts.distribution`**: Uses `DistributionOutput` for the forecasting head.

## Connection Flow
1.  **Input Scaling**: Data is standardized using `PackedStdScaler`.
2.  **Two-Stage Projection**:
    - `in_proj` maps patches to representations.
    - `feat_proj` further processes these representations before the encoder.
3.  **MoE Encoding**: The representations pass through Transformer layers where tokens are routed to a subset of experts.
4.  **Distribution Mapping**: The final hidden states are projected to distribution parameters.
5.  **Forecasting**: The model produces a predictive distribution used for sampling or NLL calculation.
6.  **Inference**: `MoiraiMoEForecast` orchestrates the process, picking patch sizes and generating samples.
