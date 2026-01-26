# Distribution Module

## Overview

The distribution module provides probabilistic output heads for MOIRAI time series forecasting models. It defines parametric distributions used for generating forecasts with uncertainty estimates, including mixture distributions for complex data patterns.

## Core Architecture

### DistributionOutput Base Class (`_base.py`)

**Abstract Base Class:**
- `DistributionOutput`: Abstract class defining interface for all distribution types
- Requires subclasses to implement `args_dim` and `domain_map` properties
- Provides automatic parameter projection layer generation

**Key Components:**
- `DistrParamProj`: Converts model representations to distribution parameters
- `AffineTransformed`: Applies scale/location transformations to distributions
- `tree_map_multi`, `convert_to_module/container`: Utilities for nested parameter handling

**Design Patterns:**
- Abstract property enforcement using `@abstract_class_property`
- PyTree-based parameter structures for complex distributions
- Domain mapping for constraining parameters to valid ranges

### Distribution Types

**Parametric Distributions:**
- **Normal**: `NormalOutput` - Standard Gaussian distribution
- **StudentT**: `StudentTOutput` - Heavy-tailed distribution for outliers
- **LogNormal**: `LogNormalOutput` - Right-skewed positive values
- **Pareto**: `ParetoOutput` - Power-law distributions
- **NegativeBinomial**: `NegativeBinomialOutput` - Count data
- **Laplace**: `LaplaceOutput` - Symmetric heavy-tailed

**Special Classes:**
- **NormalFixedScale**: `NormalFixedScaleOutput` - Normal with fixed variance
- **Mixture**: `MixtureOutput` - Weighted mixture of component distributions

### Mixture Distribution (`mixture.py`)

**Mixture Class:** Custom implementation of mixture distributions.

**Key Features:**
- Supports arbitrary component distributions
- Efficient log-probability computation without NaN gradients
- Proper mean and variance calculations using law of total variance
- Categorical weight sampling for component selection

**MixtureOutput:** Distribution output for mixture heads in MOIRAI models.

## MOIRAI Output Structure

MOIRAI uses four primary distributions in its mixture output head:

1. **Student's t-distribution** (`StudentTOutput`)
   - Handles heavy-tailed data and outliers
   - Args: `df` (degrees of freedom), `loc`, `scale`

2. **Negative Binomial** (`NegativeBinomialOutput`) 
   - Specialized for count data and sales forecasting
   - Args: `mean`, `dispersion`

3. **Log-normal** (`LogNormalOutput`)
   - For right-skewed positive data (economic indicators, sizes)
   - Args: `loc`, `scale`

4. **Normal** (`NormalOutput`)
   - General-purpose distribution for most forecasting tasks
   - Args: `loc`, `scale`

The mixture distribution learns appropriate weights for combining these four distributions based on the input data characteristics.

## Distribution Parameter Handling

### Domain Mapping
Each distribution provides `domain_map` functions to transform unconstrained neural network outputs to valid parameter ranges:

```python
@property
def domain_map(self) -> PyTree[Callable]:
    return dict(
        loc=lambda x: x,  # Identity
        scale=lambda x: F.softplus(x).clamp_min(eps)  # Positive
    )
```

### Parameter Projection
`get_param_proj()` creates projection layers from model representations to distribution parameters, supporting:
- Single dimension outputs
- Multi-dimensional parameter sets
- Variable output feature counts

## Relationship to Other Modules

### With Model Module
- Distribution outputs are used as the final layer of MOIRAI models
- `MixtureOutput` with 4 components is the standard setup
- Parameter projection integrated with transformer outputs

### With Common Module
- `jaxtyping` for tensor shape annotations in distribution classes
- `torch_util` functions used in mixture sampling

### With Training Losses
- Distribution objects used in negative log-likelihood loss computation
- Mixture distributions enable proper uncertainty quantification

## Key Design Decisions

### Mixture Distribution Choice
- **Student's t**: Robust to outliers in financial/economic data
- **Negative Binomial**: Handles discrete counts and seasonal spikes  
- **Log-normal**: Appropriate for multiplicative processes
- **Normal**: Baseline distribution for regular patterns

### Parameter Constraints
- All distributions use domain mapping to ensure parameter validity
- Softplus activation for positive parameters (scales, dispersions)
- Identity mapping for unconstrained parameters (locations)

### PyTree Structure
- Nested parameter structures enable complex distributions
- Consistent interface across simple and mixture distributions
- Support for arbitrary component combinations

### Efficiency Considerations
- Mixture distribution avoids NaN gradients through careful masking
- PyArrow acceleration where possible
- Batching optimized for transformer architectures

## Usage Examples

### Standard MOIRAI Mixture
```python
from uni2ts.distribution import (
    StudentTOutput, NegativeBinomialOutput, 
    LogNormalOutput, NormalOutput, MixtureOutput
)

components = [
    StudentTOutput(),
    NegativeBinomialOutput(), 
    LogNormalOutput(),
    NormalOutput()
]
output_dist = MixtureOutput(components)

# Get projection layer
param_proj = output_dist.get_param_proj(in_features=512, out_features=1)
```

### Individual Distribution
```python
from uni2ts.distribution import NormalOutput

normal_out = NormalOutput()
params = normal_out.get_param_proj(512, 1)(transformer_output)
dist = normal_out.distribution(params)
```

This module provides the probabilistic foundation for MOIRAI's uncertainty-aware forecasting capabilities.