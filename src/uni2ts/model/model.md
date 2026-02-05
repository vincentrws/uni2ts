# Model Module

The `model` module contains the core time series foundation model implementations. It follows a hierarchical structure where the base neural network architectures are defined in `module.py` files, and orchestration for pretraining, fine-tuning, and forecasting is provided in separate scripts.

## Files & Sub-modules

- [`moirai/`](moirai/moirai.md): Implementation of the original MOIRAI Masked Encoder Transformer.
- [`moirai_moe/`](moirai_moe/moirai_moe.md): Mixture-of-Experts variant for increased capacity.
- [`moirai2/`](moirai2/moirai2.md): Second generation MOIRAI with simplified architecture and quantile regression.

## Architecture Philosophy

The models in this module are designed with several key principles:
1.  **Universality**: Ability to handle any number of variates and any frequency.
2.  **Modularity**: Reusing components from `uni2ts.module` (Transformer blocks, specialized attention, etc.).
3.  **Efficiency**: Built to work with packed sequences and optimized scaling.
4.  **Probabilistic Output**: Moving beyond point estimates to provide full distributions or quantiles.

## Inter-dependencies
- **`uni2ts.module`**: Provides the building blocks (attention, FFN, norm, ts_embed).
- **`uni2ts.distribution`**: Used by MOIRAI 1.0 variants for parametric output heads.
- **`uni2ts.transform`**: Essential for mapping raw data entries to the tensor format expected by the models.
- **GluonTS**: Used via wrappers in `forecast.py` to provide a standard interface for time series inference.

## Connection Flow
1.  **Input**: Transformed dictionary containing tensors like `target`, `observed_mask`, `time_id`, etc.
2.  **Backbone**: Data passes through the specific Transformer variant (Encoder or MoE).
3.  **Head**: Hidden states are projected to the target dimension (distribution parameters or quantiles).
4.  **Result**: A probabilistic distribution or a set of quantile forecasts is returned.
