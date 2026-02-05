# Data Indexer Module

The `indexer` module provides an abstraction for accessing data from various underlying storage formats. It is primarily used to provide a consistent interface for retrieving individual time series or batches of time series from datasets.

## Files

- [`_base.py`](_base.py): Defines the abstract `Indexer` base class.
- [`hf_dataset_indexer.py`](hf_dataset_indexer.py): Implementation for indexing Hugging Face `Dataset` objects.

## Key Classes

### `Indexer` (in `_base.py`)

An abstract base class that inherits from `collections.abc.Sequence`.

#### Purpose
Defines the interface for all indexers, allowing them to be used like standard Python sequences (supporting `len()` and `__getitem__`).

#### Methods
- `__getitem__(idx)`: Handles integer, slice, and iterable indexing, delegating to internal methods.
- `get_uniform_probabilities()`: Returns a uniform probability distribution over all items in the dataset.
- `get_proportional_probabilities(field)`: Returns a probability distribution proportional to the length of the time series in the specified field.

### `HuggingFaceDatasetIndexer` (in `hf_dataset_indexer.py`)

#### Purpose
A specialized indexer for Hugging Face datasets. It optimizes data retrieval by leveraging `pyarrow` for fast column-wise access and conversion to NumPy arrays.

#### Features
- **Column Separation**: Automatically separates sequence columns (like `target`) from non-sequence columns.
- **PyArrow Optimization**: Uses `pyarrow` and `pyarrow.compute` for efficient slicing and length calculation.
- **Fast Proportional Sampling**: Uses `pc.list_value_length` to quickly compute lengths of all time series in a dataset without loading the data into memory.

## Inter-dependencies
- **Hugging Face `datasets`**: The target storage format for this implementation.
- **PyArrow**: Used for high-performance data access.
- **NumPy**: The primary output format for retrieved data.
- **`uni2ts.common.typing`**: Uses shared data types like `UnivarTimeSeries` and `MultivarTimeSeries`.

## Connection Flow
1. **Instantiation**: A `DatasetBuilder` creates a `HuggingFaceDatasetIndexer` for a specific dataset on disk.
2. **Access**: A `TimeSeriesDataset` uses the indexer to retrieve raw data samples.
3. **Sampling**: If a sampler is used (e.g., for multi-series datasets), it uses `get_proportional_probabilities` to decide which series to sample from.
4. **Retrieval**: When an index is requested, the indexer queries the underlying `pyarrow` table, converts the results to NumPy, and returns a dictionary of arrays.
5. **Transformation**: The retrieved data is then passed to the transformation pipeline.
