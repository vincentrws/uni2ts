# Distribution Module

The `distribution` module provides a flexible framework for probabilistic forecasting. It maps neural network outputs (representations) to parameters of various probability distributions, enabling the model to express uncertainty in its predictions.

## Files

- [`_base.py`](_base.py): Core abstractions, including `DistributionOutput` and `DistrParamProj`.
- [`mixture.py`](mixture.py): Implementation of mixture distributions (combining multiple base distributions).
- [`normal.py`](normal.py): Normal (Gaussian) distribution output.
- [`student_t.py`](student_t.py): Student's T-distribution output (robust to outliers).
- [`laplace.py`](laplace.py): Laplace distribution output.
- [`log_normal.py`](log_normal.py): Log-Normal distribution output (for positive-only data).
- [`negative_binomial.py`](negative_binomial.py): Negative Binomial distribution output (for count data).
- [`pareto.py`](pareto.py): Pareto distribution output (for heavy-tailed data).

## Key Classes

### `DistributionOutput` (in `_base.py`)

An abstract base class for defining distribution-based outputs.

#### Methods
- `distribution(distr_params, loc, scale)`: Creates a PyTorch `Distribution` object from parameters. It supports optional affine transformations (loc/scale).
- `get_param_proj(in_features, out_features)`: Returns a `DistrParamProj` layer tailored for the specific distribution's parameters.

### `DistrParamProj` (in `_base.py`)

#### Purpose
A neural network module that projects hidden representations to the unconstrained parameter space of a distribution, and then applies a `domain_map` to ensure parameters are within valid ranges (e.g., positive variance).

### `Mixture` & `MixtureOutput` (in `mixture.py`)

#### Purpose
Allows the model to predict a weighted combination of several different distributions. This is extremely powerful for handling multimodal or complex empirical data distributions.
- `MixtureOutput` takes a list of other `DistributionOutput` instances and adds a set of logits for the mixture weights.

## Distribution Parameters & Domains

Each distribution implementation defines:
- `args_dim`: The number of parameters (e.g., `{"loc": 1, "scale": 1}`).
- `domain_map`: Functions to constrain outputs (e.g., using `F.softplus` for scales to ensure they are positive).

## Inter-dependencies
- **PyTorch Distributions**: Wraps and extends standard PyTorch distributions.
- **`uni2ts.module.ts_embed`**: Uses `MultiOutSizeLinear` for projections.
- **Jaxtyping**: Used for shape and type annotations of parameters.

## Connection Flow
1. **Model Output**: The Transformer encoder produces a hidden representation tensor.
2. **Projection**: `DistrParamProj` (created by the specific `DistributionOutput`) projects these representations to raw parameter values.
3. **Domain Mapping**: Raw values are passed through `domain_map` (e.g., Softplus) to get valid distribution parameters.
4. **Distribution Creation**: `DistributionOutput.distribution()` instantiates the final PyTorch `Distribution` object.
5. **Loss/Inference**:
    - During training: The `log_prob` of the target values is computed for the loss.
    - During inference: Samples are drawn from the distribution to generate forecasts.
