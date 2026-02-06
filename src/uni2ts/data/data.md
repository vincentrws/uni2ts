# Data Module

## Overview

The data module provides comprehensive data handling infrastructure for MOIRAI time series forecasting. It includes dataset loading, batching, indexing, and builders for processing time series data from various formats into model-ready inputs.

## Components

### Dataset Classes (`dataset.py`)

**TimeSeriesDataset**: Base dataset class for time series data.

**Key Features:**
- Supports different sampling strategies (uniform, proportional, none)
- Applies transformations to raw data
- Flexible dataset weighting for balancing

**MultiSampleTimeSeriesDataset**: Samples and stacks multiple time series.
- Combines multiple aligned time series into single samples
- Configurable fields to combine (e.g., targets)
- Beta-binomial sampling for variable numbers of series

**EvalDataset/FinetuneDataset**: Specialized for evaluation and fine-tuning.
- Sliding/rolling window splitting
- Validation split support
- Window-based sampling for evaluation

### Data Loading (`loader.py`)

**Collation Classes:**
- `Collate`: Abstract base for collation strategies
- `PadCollate`: Pads variable-length sequences with configurable padding
- `PackCollate`: Uses first-fit decreasing bin packing for efficient batching

**DataLoader Class:**
- Enhanced PyTorch DataLoader with sequence packing
- Cycle support for infinite training
- Fixed batch size enforcement even with batch factors
- Management of batched sample queues and padding

**Data Structures:**
- `SliceableBatchedSample`: Enables slicing of batched tensors
- `BatchedSampleQueue`: Efficient queue for batch management with schema validation

### Indexing System (`indexer/`)

**Base Indexer (`_base.py`):**
- Abstract sequence interface for data indexing
- Validation of indices and slices
- Automatic probability distribution computation for sampling

**HuggingFaceDatasetIndexer (`hf_dataset_indexer.py`):**
- Integration with Hugging Face datasets
- Efficient PyArrow operations for data access
- Unified handling of univariate/multivariate time series

### Dataset Builders (`builder/`)

**Base Classes (`_base.py`):**
- `DatasetBuilder`: Abstract interface for building/loading datasets
- `ConcatDatasetBuilder`: Combines multiple dataset builders

**Simple Builders (`simple.py`):**
- `SimpleDatasetBuilder`: Builds datasets from CSV files (long/wide formats)
- `SimpleFinetuneDatasetBuilder`: Specialized for fine-tuning with normalization
- `SimpleEvalDatasetBuilder`: Evaluation dataset builder

**Functions:**
- `get_finetune_builder()`, `get_eval_builder()`: Convenience functions for standard configurations
- CSV parsing with frequency inference, normalization support
- HuggingFace dataset integration for storage

### Relationship to Other Modules

**With Common:**
- Uses `typing.py` for type definitions
- Leverages `torch_util.py` for tensor operations in collation
- Environment paths from `Env` class

**With Transform:**
- Transformations applied via transform chains
- Dataset classes couple directly with transformation pipelines

**With Model:**
- Provides properly batched and packed sequences
- Integrates with PyTorch Lightning training loops

**With Training:**
- Datasets used in Lightning DataModules
- Collation strategies optimized for transformer attention patterns

## Key Design Decisions

### Sequence Packing
- Reduces padding from ~61% to <0.4%
- Enables larger effective batch sizes
- Implemented via PackCollate with first-fit decreasing bin packing

### Batched Sample Management
- Complex batching logic abstracted into queue structures
- Supports fixed-size batching with padding/filling strategies
- Schema validation prevents mismatched data formats

### Modular Indexing
- Pluggable indexer system supports different storage backends
- Base class ensures consistent interface
- PyArrow integration for efficient data access

### Builder Pattern
- Clean separation between data processing and dataset creation
- Reusable builders for different data formats
- HuggingFace integration for scalable storage

## Usage Patterns

### Training Dataset
```python
from uni2ts.data.dataset import TimeSeriesDataset
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import patch_transform

dataset = TimeSeriesDataset(
    indexer=HuggingFaceDatasetIndexer(hf_dataset),
    transform=patch_transform,
    sample_time_series_type.UNIFORM,
    dataset_weight=1.0
)
```

### Packed DataLoader
```python
from uni2ts.data.loader import DataLoader, PackCollate

loader = DataLoader(
    dataset=dataset,
    batch_size=32,
    collate_fn=PackCollate(seq_fields=['target'], max_length=512)
)
```

This module provides a complete end-to-end data pipeline from raw files to model-ready batches, optimized for MOIRAI's transformer-based architecture.