# Packed Loss Module

The `loss.packed` module provides loss function implementations designed to work with packed time series sequences. These functions correctly handle masking and normalization across multiple sequences that have been packed into a single batch entry.

## Files

- [`_base.py`](_base.py): Defines abstract base classes for packed losses and the core logic for loss reduction.
- [`distribution.py`](distribution.py): Loss functions for probabilistic forecasts (e.g., Negative Log-Likelihood).
- [`point.py`](point.py): Loss functions for point forecasts (e.g., MSE, MAE).
- [`quantile.py`](quantile.py): Loss functions for quantile forecasts (e.g., Pinball loss).
- [`normalized.py`](normalized.py): Normalized versions of standard losses.
- [`percentage_error.py`](percentage_error.py): Percentage-based error metrics.

## Key Classes

### `PackedLoss` (in `_base.py`)

An abstract base class for all losses supporting packed sequences.

#### Purpose
To compute loss while respecting `prediction_mask` (only training on future/masked values), `observed_mask` (ignoring missing data), and `sample_id` (separating packed sequences).

#### Methods
- `__call__(...)`: Orchestrates the loss computation and reduction.
- `_loss_func(...)`: Abstract method implemented by subclasses to compute the per-token loss.
- `reduce_loss(...)`: Performs a complex reduction that normalizes the loss correctly. It ensures that each token's contribution is scaled by the total number of observed prediction tokens in its respective sequence (`tobs`) and the total number of sequences (`nobs`).

### `PackedNLLLoss` (in `distribution.py`)

#### Purpose
Computes the Negative Log-Likelihood (NLL) for a predictive distribution. It is the primary loss function used during MOIRAI pretraining.

## Inter-dependencies
- **PyTorch**: Used for all tensor operations.
- **`uni2ts.common.torch_util`**: Uses `safe_div` for normalization.
- **Jaxtyping**: Extensive use of shape and type annotations for complex batch/sequence/dimension tensors.

## Connection Flow
1.  **Forward Pass**: The model produces a distribution or point estimate for a packed batch.
2.  **Loss Calculation**: The `PackedLoss` instance is called with the prediction and metadata (`sample_id`, masks).
3.  **Per-Token Loss**: `_loss_func` computes the raw loss (e.g., `-distr.log_prob(target)`).
4.  **Reduction**: `reduce_loss` applies masks and computes a weighted average that accounts for sequence packing, ensuring that sequences of different lengths contribute fairly to the gradient.
5.  **Optimization**: The resulting scalar loss is used by PyTorch Lightning to perform backpropagation.
