# Indexer Module

## Overview

The indexer module provides a unified interface for random access to time series datasets in MOIRAI. It abstracts away the underlying storage format (HuggingFace datasets, etc.) and provides efficient indexing capabilities for training and inference. The module supports individual item access, batched retrieval, and probability-based sampling for dataset weighting.

## Components

### Base Classes (`_base.py`)

**Key Classes:**
- `Indexer`: Abstract base class implementing the Sequence protocol for dataset indexing

**Functionality:**
- Implements standard Python sequence behavior (`__len__`, `__getitem__`)
- Supports indexing by integer, slice, and iterable (for batched access)
- Provides probability calculation methods for uniform and proportional sampling
- Encapsulates length validation and bounds checking

**Data Access Methods:**
- `_getitem_int()`: Retrieve single item by index
- `_getitem_slice()`: Retrieve batch via slice
- `_getitem_iterable()`: Retrieve batch via arbitrary indices

### HuggingFace Dataset Indexer (`hf_dataset_indexer.py`)

**Key Classes:**
- `HuggingFaceDatasetIndexer`: Concrete implementation for HuggingFace datasets

**Functionality:**
- Wraps HuggingFace Dataset objects with efficient indexing
- Uses PyArrow for high-performance sequence column access
- Handles both sequence and non-sequence features
- Supports uniform and variable-length datasets

**Core Methods:**
- `__len__()`: Returns total number of samples
- `_pa_column_to_numpy()`: Efficiently converts PyArrow arrays to NumPy
- `get_proportional_probabilities()`: Fast length-based probability calculation using PyArrow compute

## Data Retrieval Patterns

### Single Item Access
```python
indexer = HuggingFaceDatasetIndexer(dataset)
item = indexer[42]  # Returns dict with 'item_id', 'start', 'freq', 'target', etc.
```

### Batch Access via Slice
```python
batch = indexer[0:32]  # Returns batched data for first 32 samples
```

### Arbitrary Index Access
```python
indices = [5, 12, 25, 33]
batch = indexer[indices]  # Returns batch with specified indices
```

## Probability and Sampling

### Uniform Sampling
- All time series have equal probability
- Useful when dataset balance is already appropriate

### Proportional Sampling
- Probability weighted by time series length
- Longer series have higher probability of selection
- Useful for training efficiency and memory utilization

### Usage Example
```python
# Get probabilities for sampling
probs = indexer.get_proportional_probabilities()

# Sample using torch or numpy
import torch
indices = torch.multinomial(torch.tensor(probs), num_samples=32, replacement=True)
batch = indexer[indices]
```

## Relationship to Other Modules

### With Data Module
- Indexers are used by `TimeSeriesDataset`, `FinetuneDataset`, and `EvalDataset` classes
- Provide the raw indexed data that transforms operate on
- Enable efficient data loading during training loops

### With Builder Module
- Builders create HuggingFace datasets on disk
- Indexers wrap those datasets for training-time random access
- Builders → Indexers pipeline enables disk caching and fast iterative access

### With Transform Module
- Indexed data passes through transformation chains
- Transforms expect indexer output format
- PyTorch dataloaders consume transformed indexed data

## Design Pattern

The indexer module follows the Adapter pattern:

- **Abstract Interface**: Indexer base class defines standard access protocol
- **Concrete Adaptations**: Specific indexers adapt different data formats to the interface
- **Protocol Compliance**: Sequence protocol ensures compatibility with Python's data access expectations
- **Performance Optimization**: PyArrow integration provides C++ speed for data retrieval

This design enables:
- Consistent data access across different storage backends
- Easy extension to new dataset formats
- Efficient batching and sampling operations
- Clean separation between data storage and data access

## Performance Considerations

### PyArrow Integration
- Zero-copy data transfer where possible
- Vectorized operations for sequence length calculations
- Arrow's columnar format provides efficient access patterns

### Memory Management
- Lazy loading prevents memory bloat
- Batch access reduces function call overhead
- Uniform length optimization skips redundant length checks

### Sampling Efficiency  
- PyArrow compute operations for fast proportional probability calculation
- Avoids loading full datasets into memory during probability computation
- Supports large-scale datasets through streaming operations
