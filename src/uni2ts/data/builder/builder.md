# Data Builder Module

The `builder` module provides an abstraction for building and loading time series datasets. It handles the conversion of various data formats into a unified format (typically Hugging Face `datasets`) and facilitates their loading into PyTorch `Dataset` objects.

## Files

- [`_base.py`](_base.py): Defines the abstract `DatasetBuilder` base class and concatenation utilities.
- [`simple.py`](simple.py): Implements builders for common data formats like CSV/DataFrames (long and wide formats).
- [`lotsa_v1/`](lotsa_v1/lotsa_v1.md): Sub-module for building the LOTSA V1 dataset collection.

## Key Classes

### `DatasetBuilder` (in `_base.py`)

An abstract base class.

#### Methods
- `build_dataset(*args, **kwargs)`: Abstract method to convert raw data into the library's internal storage format (e.g., Hugging Face datasets on disk).
- `load_dataset(transform_map)`: Abstract method to load the preprocessed data from disk into a PyTorch-compatible `Dataset`.

### `ConcatDatasetBuilder` (in `_base.py`)

#### Purpose
Allows multiple `DatasetBuilder` instances to be treated as a single builder. When `load_dataset` is called, it loads all sub-datasets and returns a `ConcatDataset`.

### `SimpleDatasetBuilder` (in `simple.py`)

#### Purpose
A practical implementation for loading datasets from CSV files or Pandas DataFrames. It supports both "long" and "wide" (univariate or multivariate) formats.

#### Methods
- `build_dataset(file, dataset_type, ...)`: Reads a CSV, converts it to a Hugging Face dataset, and saves it to disk.
- `load_dataset(transform_map)`: Loads the saved dataset from disk and wraps it in a `TimeSeriesDataset`.

### `SimpleFinetuneDatasetBuilder` & `SimpleEvalDatasetBuilder` (in `simple.py`)

#### Purpose
Specialized builders for fine-tuning and evaluation, respectively. They support normalization and creating sliding/rolling windows for evaluation.

## Inter-dependencies
- **Hugging Face `datasets`**: The primary format for intermediate data storage.
- **Pandas**: Used for initial data manipulation in `SimpleDatasetBuilder`.
- **PyTorch**: Subclasses use `Dataset` and `ConcatDataset`.
- **`uni2ts.data.dataset`**: Uses `TimeSeriesDataset`, `EvalDataset`, and `FinetuneDataset`.
- **`uni2ts.data.indexer`**: Uses `HuggingFaceDatasetIndexer` for efficient access.

## Connection Flow
1. **Preprocessing**: Raw data (CSV, GluonTS, etc.) is processed by a `build_dataset` call, resulting in a directory on disk.
2. **Configuration**: The training/eval script defines which builders to use via Hydra.
3. **Loading**: `load_dataset` is called, which:
    - Loads the dataset from disk.
    - Creates an indexer.
    - Applies transformations from the `transform_map`.
    - Returns a `Dataset` object ready for the `DataLoader`.
4. **Batching**: If multiple builders are used via `ConcatDatasetBuilder`, their outputs are concatenated for batching.
