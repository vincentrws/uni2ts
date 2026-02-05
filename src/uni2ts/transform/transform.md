# Transform Module

The `transform` module implements a sophisticated data preprocessing pipeline. It uses a chainable transformation pattern to convert raw GluonTS-style data entries into the structured tensors required by the MOIRAI models.

## Files

- [`_base.py`](_base.py): Defines the base `Transformation` class and `Chain` utility.
- [`patch.py`](patch.py): Logic for time series patching and patch size selection.
- [`feature.py`](feature.py): Adds temporal, variate, and sample indices to data.
- [`reshape.py`](reshape.py): Handles tensor packing, transposing, and sequence formatting.
- [`task.py`](task.py): Implements masked prediction logic for training.
- [`crop.py`](crop.py): Utilities for cropping time series into manageable lengths.
- [`pad.py`](pad.py): Utilities for padding sequences.
- [`imputation.py`](imputation.py): Handles missing value imputation.
- [`resample.py`](resample.py), [`field.py`](field.py): Other data manipulation transforms.

## Key Classes

### `Transformation` (in `_base.py`)

An abstract base class. Transformations are callables that take a `dict[str, Any]` and return a modified `dict[str, Any]`. They can be combined using the `+` operator (e.g., `Transform1() + Transform2()`).

### `GetPatchSize` (in `patch.py`)

#### Purpose
Determines the patch size for a time series based on its frequency and length. It ensures the resulting sequence has at least a minimum number of patches.

### `AddVariateIndex` & `AddTimeIndex` (in `feature.py`)

#### Purpose
Assigns unique IDs to different variates and time steps.
- `variate_id`: Used by `BinaryAttentionBias` to capture relationships between different variables.
- `time_id`: Used by RoPE to capture temporal order.

### `MaskedPrediction` (in `task.py`)

#### Purpose
Prepares the "Masked Encoder" training task. It randomly selects a portion of the time series to be masked (the "prediction horizon") and generates a `prediction_mask`.

### `PackFields` & `FlatPackCollection` (in `reshape.py`)

#### Purpose
Organizes various fields (targets, masks, IDs) into the final packed format. These transforms use `einops.pack` to merge list-of-arrays into multi-dimensional tensors suitable for batch collation.

## Inter-dependencies
- **GluonTS**: Transformations often work with GluonTS-style `DataEntry` structures.
- **NumPy**: The primary engine for data manipulation inside transforms.
- **Einops**: Used extensively for reshaping and packing operations.
- **`uni2ts.common.typing`**: Uses shared time series type definitions.

## Connection Flow
1.  **Dataset Load**: `TimeSeriesDataset` retrieves a raw entry.
2.  **Chain Execution**: The entry is passed through a `Chain` of transformations.
3.  **Indexing**: `AddTimeIndex` and `AddVariateIndex` add metadata.
4.  **Patching**: `Patchify` converts the series into patches of size $P$.
5.  **Task Preparation**: `MaskedPrediction` decides which patches to hide from the encoder.
6.  **Formatting**: `PackFields` merges everything into a structured dictionary.
7.  **Collation**: The transformed dictionary is then ready for the `DataLoader`'s collation step (where sequence packing happens).
8.  **Model**: Finally, the model receives a batch of these transformed dictionaries.
