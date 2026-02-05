# Common Module

The `common` module provides core utilities, environment management, configuration helpers, and shared type definitions used throughout the `uni2ts` library.

## Files

- [`core.py`](core.py): Fundamental Python utilities.
- [`env.py`](env.py): Environment variable and path management.
- [`hydra_util.py`](hydra_util.py): Custom Hydra resolvers and configuration utilities.
- [`torch_util.py`](torch_util.py): PyTorch-specific utility functions, especially for handling packed sequences.
- [`typing.py`](typing.py): Shared type aliases and custom JAX-typing dtypes.
- [`sampler.py`](sampler.py): (Not read, but likely contains sampling utilities)

## Key Components

### `core.py`
- `abstract_class_property(*names)`: A decorator for classes that ensures subclasses define specific attributes. It uses `__init_subclass__` to enforce these requirements at definition time.

### `env.py`
- `Env` class: A singleton that loads environment variables from a `.env` file using `python-dotenv`.
- Path Variables: Manages paths like `LOTSA_V1_PATH`, `LSF_PATH`, and `HF_CACHE_PATH`.

### `hydra_util.py`
- Custom Hydra Resolvers:
    - `as_tuple`: Converts a list to a tuple in Hydra configs.
    - `cls_getattr`: Dynamically retrieves an attribute from a class.
    - `floordiv`: Performs floor division.
    - `mul`: Performs multiplication.

### `torch_util.py`
- **Packed Attention Masks**:
    - `packed_attention_mask`: Generates an attention mask that allows tokens within the same packed sequence to attend to each other but prevents cross-sequence attention.
    - `packed_causal_attention_mask`: Adds causal masking to the packed attention mask.
- **Masking & Padding**:
    - `mask_fill`: Fills masked portions of a tensor with a specific value.
    - `size_to_mask`: Converts sequence sizes to binary masks.
- **Aggregations**:
    - `sized_mean`, `masked_mean`: Compute means while respecting variable sequence lengths or binary masks.
- `safe_div`: Handles division by zero by substituting 1.0 in the denominator.

### `typing.py`
- Defines custom dtypes for `jaxtyping` such as `DateTime64` and `Character`.
- Shared aliases for time series data: `UnivarTimeSeries`, `MultivarTimeSeries`.
- Aliases for samples: `Sample`, `BatchedSample`.

## Inter-dependencies
- **Hydra/OmegaConf**: `hydra_util.py` integrates with the configuration system.
- **PyTorch**: `torch_util.py` provides essential tensor operations.
- **Jaxtyping**: Used extensively for type-safe tensor annotations.
- **python-dotenv**: Used for environment management.

## Connection Flow
1. **Startup**: `Env` is initialized, loading paths used by data builders and loaders.
2. **Configuration**: Hydra uses resolvers in `hydra_util.py` to process complex configuration files.
3. **Data Processing**: `typing.py` ensures consistent data structures across builders and transforms.
4. **Model Execution**: `torch_util.py` functions are used inside model forward passes (especially in `MoiraiModule`) to handle packed sequences and compute losses.
