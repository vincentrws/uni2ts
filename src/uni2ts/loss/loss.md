# Loss Module

The `loss` module defines the optimization objectives for the MOIRAI models. It is designed to be extensible and specifically optimized for the sequence packing training strategy.

## Files

- [`packed/`](packed/packed.md): Contains loss functions specifically designed for packed sequences, handling multiple time series within a single batch entry.

## Summary

The primary focus of this module is on "packed" losses. Because MOIRAI uses sequence packing to improve training efficiency, standard PyTorch loss functions cannot be used directly as they would aggregate loss over padding tokens and fail to distinguish between different sequences in a single packed batch.

The implementations in [`loss.packed`](packed/packed.md) address this by using `sample_id` and `prediction_mask` to ensure that gradients are only computed for valid, masked tokens and are normalized correctly per sequence.

## Key Concepts
- **Masked Prediction Loss**: Training only on tokens that are part of the prediction horizon or intentionally masked.
- **Probabilistic Loss**: Using Negative Log-Likelihood (NLL) of parametric distributions.
- **Point & Quantile Loss**: Support for deterministic forecasting and quantile regression.
- **Reduction Strategy**: Custom normalization that handles variable lengths and sequence packing to ensure stable training across diverse datasets.
