# Common Module

## Overview

The common module provides shared utilities, constants, and base classes used throughout the MOIRAI codebase. It contains core functionality for inheritance patterns, environment management, configuration utilities, sampling, PyTorch utilities, and type definitions.

## Components

### Core (`core.py`)

**Key Classes:**
- `abstract_class_property()`: Decorator that enforces subclasses to implement specified class attributes or properties.

**Functionality:**
- Uses `__init_subclass__` hook to check that required attributes are defined in child classes
- Raises `NotImplementedError` with descriptive messages when requirements aren't met
- Works with multiple attribute names passed as arguments

### Environment (`env.py`)

**Key Classes:**
- `Env`: Singleton class for managing environment variables and configuration paths.

**Functionality:**
- Loads `.env` file using `dotenv`
- Provides access to key dataset paths: `LOTSA_V1_PATH`, `LSF_PATH`, `CUSTOM_DATA_PATH`, `HF_CACHE_PATH`
- Monkey-patches path attributes as class attributes for easy access
- Warns if `.env` file cannot be loaded

**Usage:**
```python
from uni2ts.common import env
path = env.CUSTOM_DATA_PATH
```

### Hydra Utilities (`hydra_util.py`)

**Key Functions:**
- `register_resolver()`: Decorator to register custom OmegaConf resolvers
- `resolve_as_tuple()`: Convert list to tuple in configs
- `resolve_cls_getattr()`: Dynamic attribute access on classes for configs
- `resolve_floordiv()`: Floor division operation in configs
- `resolve_mul()`: Multiplication operation in configs

**Purpose:**
- Extends Hydra configuration system with additional mathematical and type operations
- Enables more flexible configuration workflows
- Supports dynamic class attribute resolution for cleaner config files

### Samplers (`sampler.py`)

**Key Functions:**
- `uniform_sampler()`, `binomial_sampler()`, `beta_binomial_sampler()`: Different probability distributions for sampling integers
- `get_sampler()`: Factory function to create samplers by distribution name

**Purpose:**
- Provides configurable sampling strategies for dynamic sequence lengths in training
- Used for varying context lengths, prediction horizons, and sequence packing
- `beta_binomial_sampler` provides flexible length distributions (defaults to uniform when a=b=1)

### Torch Utilities (`torch_util.py`)

**Key Functions:**
- `packed_attention_mask()`: Creates attention masks for packed sequences based on sample IDs
- `packed_causal_attention_mask()`: Creates causal attention masks within samples
- `mask_fill()`: Efficient masking and filling operations
- `safe_div()`: Division that handles zero denominators
- `size_to_mask()`: Converts sizes to boolean masks
- `sized_mean()`, `masked_mean()`: Mean calculations with proper masking
- `unsqueeze_trailing_dims()`: Tensor shape manipulation

**Purpose:**
- Essential utilities for working with packed sequences in transformers
- Efficient attention computation for variable-length sequences
- Masks and operations that handle the batching scheme used in sequence packing

**Type Dictionary:**
- `numpy_to_torch_dtype_dict`: Mapping from NumPy dtypes to PyTorch dtypes

### Typing (`typing.py`)

**Custom Types:**
- `DateTime64`: Abstract dtype for datetime64 arrays
- `Character`: Abstract dtype for string arrays

**Type Aliases:**
- Data containers: `DateTime`, `String`, `BatchedDateTime`, `BatchedString`
- Time series types: `UnivarTimeSeries`, `MultivarTimeSeries`, `BatchedData`
- Model inputs: `Sample`, `BatchedSample`
- Generators: `GenFunc`, `SliceableGenFunc`

**Purpose:**
- Comprehensive type annotations using `jaxtyping` for tensor shapes and dtypes
- Clear contracts for data formats throughout the pipeline
- Compile-time type checking for data processing functions

## Relationship to Other Modules

### With Data Module
- `typing.py` provides type definitions used in data loading and processing
- `torch_util.py` provides utilities for tensor operations in data transformation
- `samplers.py` supports dynamic data sampling strategies

### With Model Module
- `torch_util.py` provides attention mask utilities crucial for transformer implementations
- `typing.py` defines input/output tensor shapes expected by models

### With Transform Module
- Shared utilities from `torch_util.py` and `typing.py`
- `abstract_class_property` used in base transformation classes

### With Training Pipeline
- `Env` manages dataset paths and training configuration
- Hydra utilities support complex experiment configurations
- Samplers enable curriculum learning through length sampling

## Design Pattern

The common module follows a utility module pattern where each file provides focused functionality used across the codebase. This avoids circular dependencies while maintaining clean interfaces through type annotations and well-defined APIs.