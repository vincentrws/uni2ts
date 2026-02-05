# Data Module

The `data` module is a core part of the `uni2ts` library, responsible for representing, loading, and batching time series data. It includes support for complex operations like sequence packing and multivariate sampling.

## Files

- [`dataset.py`](dataset.py): Defines PyTorch `Dataset` subclasses for various time series use cases.
- [`loader.py`](loader.py): Implements custom data loading logic, including sequence packing and batching strategies.
- [`builder/`](builder/builder.md): Sub-module for constructing datasets from raw files.
- [`indexer/`](indexer/indexer.md): Sub-module for efficient data access from disk.

## Key Classes

### `TimeSeriesDataset` (in `dataset.py`)

A base class inheriting from `torch.utils.data.Dataset`.

#### Purpose
Wraps an `Indexer` and applies transformations to the retrieved data. It supports different sampling strategies (none, uniform, proportional).

#### Sampling Types (`SampleTimeSeriesType`)
- `NONE`: Regular sequential indexing.
- `UNIFORM`: Samples each time series with equal probability.
- `PROPORTIONAL`: Samples time series with probability proportional to their length.

### `MultiSampleTimeSeriesDataset` (in `dataset.py`)

#### Purpose
Samples multiple time series and stacks them together. This is useful for training on related time series or for multivariate modeling where individual series are stored separately.

### `EvalDataset` & `FinetuneDataset` (in `dataset.py`)

#### Purpose
Specialized for evaluation and fine-tuning, supporting the concept of "windows" within a larger time series.

### `Collate` and its subclasses (in `loader.py`)

#### `PadCollate`
Pads uneven sequences in a batch to a fixed `max_length`.

#### `PackCollate`
Implements **Sequence Packing** using the first-fit decreasing bin packing strategy. It packs multiple short sequences into a single sequence of `max_length`, drastically reducing padding waste. It also generates a `sample_id` to let the model distinguish between different sequences in the same pack.

### `DataLoader` (in `loader.py`)

A wrapper around PyTorch's standard `DataLoader`.

#### Features
- **Packing**: Integrates with `PackCollate`.
- **Cycling**: Can infinitely repeat the dataset (useful for pretraining).
- **Fixed Batches**: Uses an internal queue (`BatchedSampleQueue`) and iterator (`_BatchedSampleIterator`) to ensure consistent batch sizes even when packing produces variable-sized outputs from the underlying collator.

## Inter-dependencies
- **PyTorch**: Built on top of `Dataset` and `DataLoader`.
- **`uni2ts.data.indexer`**: Uses indexers to fetch raw data.
- **`uni2ts.transform`**: Applies transformation chains to data before batching.
- **NumPy**: Used for sampling logic and intermediate data representation.

## Connection Flow
1. **Indexing**: `TimeSeriesDataset` requests data from the `Indexer`.
2. **Flattening**: Data is flattened into a list of univariate series.
3. **Transformation**: The `transform` chain is applied (patching, scaling, etc.).
4. **Batching**: The `DataLoader` fetches multiple samples.
5. **Collation**: `PackCollate` or `PadCollate` combines samples into a single tensor.
    - If packing, multiple samples are merged into one sequence with a shared `sample_id`.
6. **Queueing**: `_BatchedSampleIterator` ensures that the final batch sent to the model has exactly `batch_size` packs.
