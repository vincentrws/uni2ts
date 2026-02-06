# Builder Module

## Overview

The builder module provides abstraction for dataset construction and loading in MOIRAI. It converts various data sources (CSV files, benchmark datasets, etc.) into HuggingFace Dataset format, which is then indexed and used by the data pipeline for training and evaluation. The module supports both simple custom datasets and complex preprocessor datasets used in pre-training.

## Components

### Base Classes (`_base.py`)

**Key Classes:**
- `DatasetBuilder`: Abstract base class defining the builder interface with `build_dataset()` and `load_dataset()` methods
- `ConcatDatasetBuilder`: Combines multiple DatasetBuilders into a single concatenated dataset

**Functionality:**
- `DatasetBuilder` defines the contract for all concrete builders: building from source data and loading into PyTorch datasets
- `ConcatDatasetBuilder` enables composing datasets, useful for multi-dataset training
- Builders create HuggingFace datasets saved to disk, then load them back using `HuggingFaceDatasetIndexer`

### Simple Builders (`simple.py`)

**Key Classes:**
- `SimpleDatasetBuilder`: General-purpose builder for custom datasets
- `SimpleFinetuneDatasetBuilder`: Builder optimized for fine-tuning with sliding windows
- `SimpleEvalDatasetBuilder`: Builder optimized for evaluation with rolling windows

**Key Functions:**
- `_from_long_dataframe()`: Converts pandas long-format DataFrame to HuggingFace Dataset
- `_from_wide_dataframe()`: Converts pandas wide-format DataFrame to HuggingFace Dataset
- `_from_wide_dataframe_multivariate()`: Converts multivariate wide-format DataFrame to HuggingFace Dataset
- `generate_finetune_builder()`: Helper to create finetune builders with proper window calculations
- `generate_eval_builder()`: Helper to create evaluation builders with proper window calculations
- `generate_eval_builders()`: Helper to create multiple evaluation builders for parameter sweeps

**Functionality:**
- Support three data formats:
  - `long`: Each row represents one time step, with `item_id` and variable columns
  - `wide`: Columns represent time steps, each column is a univariate time series
  - `wide_multivariate`: Columns represent time steps, all columns are variables of a single multivariate time series
- Automatic frequency inference and normalization support
- Command-line interface for dataset preprocessing: `python -m uni2ts.data.builder.simple [dataset_name] [file_path] --dataset_type wide`

### LOTSA v1 Builders (`lotsa_v1/`)

**Key Classes:**
- `LOTSADatasetBuilder`: Abstract base class for LOTSA benchmark datasets
- Individual builders: `BuildingsBenchDatasetBuilder`, `Buildings900KDatasetBuilder`, `ERA5DatasetBuilder`, etc.

**Functionality:**
- Pre-processing for the LOTSA v1 benchmark datasets used in MOIRAI pre-training
- Each builder handles a specific dataset (buildings, weather, finance, etc.)
- Converts raw data sources into standardized HuggingFace format
- Command-line interface for dataset building: `python -m uni2ts.data.builder.lotsa_v1 [builder] --datasets [dataset_names]`

## Workflow

### Dataset Building Process

1. **Source Data**: Raw data files (CSV, Parquet, native formats)
2. **Builder Conversion**: Convert to HuggingFace Dataset with proper features (`item_id`, `start`, `freq`, `target`)
3. **Storage**: Save to disk in HuggingFace format
4. **Loading**: Use `HuggingFaceDatasetIndexer` for fast random access during training

### Usage Strategies

- **Custom Datasets**: Use `SimpleDatasetBuilder` for pandas-readable data
- **Benchmark Datasets**: Use LOTSA builders for standardized preprocessing
- **Multi-Dataset Training**: Use `ConcatDatasetBuilder` to combine datasets with different weights

## Relationship to Other Modules

### With Data Module
- Builders create the raw HuggingFace datasets consumed by `TimeSeriesDataset`, `FinetuneDataset`, `EvalDataset`
- Initialization includes dataset indexers (`HuggingFaceDatasetIndexer`) and transformations

### With Transform Module
- Transformation chains are applied during dataset loading via `transform_map`
- Builders create datasets with assumed structure that transformations will process

### With Training Pipeline
- Built datasets support both pre-training (on LOTSA) and fine-tuning (on custom data)
- Dataset weights enable controlling contribution of different sources during training

### With CLI
- Simple builders provide command-line interface: `python -m cli.train ... --data custom_dataset`
- LOTSA builders provide command-line interface: `python -m uni2ts.data.builder.lotsa_v1 ...`

## Design Pattern

The builder module uses the Builder pattern to encapsulate dataset construction logic:

- Abstract base classes define the interface
- Concrete builders implement specific data source conversions
- Builders are responsible for both building (from raw data) and loading (into train-ready datasets)
- Separation of concerns: builders handle data ingestion, other modules handle processing

This design enables extensibility - new datasets can be added by creating new builder classes, while the rest of the system remains unchanged.
