# Eval Util Module

## Overview

The eval_util module provides comprehensive evaluation utilities for MOIRAI time series forecasting models. It integrates with GluonTS evaluation framework and supports multiple benchmark datasets, custom datasets, and various metrics.

## Components

### Evaluation Framework (`evaluation.py`)

**Core Functions:**
- `evaluate_forecasts_raw()`: Core evaluation function returning raw metric dictionaries
- `evaluate_forecasts()`: Returns evaluation results as pandas DataFrame
- `evaluate_model()`: Evaluate a Predictor directly on test data

**Key Features:**
- Batch processing for efficient evaluation on large datasets
- Configurable axis aggregation for metrics (dataset/item/time dimensions)
- Seasonal error calculation using GluonTS utilities
- NaN handling and masking support
- Progress bars with tqdm

**BatchForecast Class:**
- Wrapper around GluonTS Forecast objects
- Adds batch dimension to individual forecast arrays
- NaN validation with configurable tolerance

### Metrics (`metrics.py`)

**Custom Metrics:**
- `MedianMSE`: Mean Squared Error using median forecast (0.5 quantile)

**Integration:**
- Extends GluonTS metric definitions
- Uses DirectMetric pattern for statistical computations
- Supports configurable aggregation axes

### Data Loading (`data.py`)

**Benchmark Dataset Loaders:**
- `get_gluonts_val_dataset()`, `get_gluonts_test_dataset()`: GluonTS benchmark datasets
- `get_lsf_val_dataset()`, `get_lsf_test_dataset()`: Long Sequence Forecasting datasets  
- `get_custom_eval_dataset()`: Custom datasets stored in Hugging Face format

**Dataset Classes:**
- `_hf_dataset.py`: HuggingFace dataset wrapper
- `_lsf_dataset.py`: Long Sequence Forecasting dataset handler
- `_pf_dataset.py`: Probabilistic Forecasting dataset repository

**MetaData Structure:**
- Standard metadata for datasets: frequency, dimensions, prediction length
- Support for dynamic features and splits

### Plotting (`plot.py`)

**Visualization Functions:**
- Forecast visualization utilities
- Comparison plots for predictions vs actuals
- Confidence interval plotting

## Key Design Patterns

### GluonTS Integration
- Leverages GluonTS's robust evaluation infrastructure
- Compatible with standard Forecast objects and Predictor interfaces
- Reuses metric definitions and data loading utilities

### Batching Strategy
- Batched evaluation for memory efficiency
- Configurable batch sizes with progress tracking
- Support for variable-length sequences

### Multi-Axis Aggregation
- Flexible metric computation across different dimensions:
  - Dataset-level aggregation
  - Item-level (per time series) metrics
  - Time-level (per time step) metrics

### Dataset Abstraction
- Unified interface across different dataset types
- Metadata-driven configuration
- Support for custom datasets and benchmarks

## Relationship to Other Modules

### With Model Module
- Evaluates Predictor objects from model creation
- Supports probabilistic forecasts with confidence intervals
- Metrics computed on distribution samples

### With Data Module
- Uses DataLoaders for batching evaluation
- Integrates with dataset splitting utilities
- Supports multiple data format conversions

### With Distribution Module
- Metrics assess distribution quality (CRPS, etc.)
- Sample-based evaluation of probabilistic forecasts
- Quantile-based metrics for uncertainty quantification

## Supported Benchmarks

### GluonTS Benchmarks
- Standard evaluation datasets (electricity, traffic, etc.)
- Pre-configured prediction lengths and frequencies
- Seasonal error calculations

### Long Sequence Forecasting (LSF)
- Dedicated benchmark for long-horizon forecasting
- Variable prediction lengths (96, 192, 336, 720 steps)
- Multi-variate and uni-variate modes

### Probabilistic Forecasting (PF)
- Focus on uncertainty quantification
- Gaussian and non-Gaussian targets
- Standardized evaluation protocols

### Custom Datasets
- HDF5-format validation splits
- Configurable windows and distances
- Integration with HuggingFace datasets

## Usage Examples

### Evaluate Forecasts
```python
from uni2ts.eval_util import evaluate_forecasts
from gluonts.ev.metrics import MASE, CRPS

metrics_df = evaluate_forecasts(
    forecasts=forecast_list,
    test_data=test_data,
    metrics=[MASE(), CRPS()],
    axis=(0, 1),  # Dataset and item aggregation
    batch_size=50
)
```

### Load Benchmark Dataset
```python
from uni2ts.eval_util.data import get_lsf_val_dataset

test_data, metadata = get_lsf_val_dataset(
    dataset_name="ETTm1", 
    prediction_length=96,
    mode="S"
)
```

### Custom Metric
```python
from uni2ts.eval_util.metrics import MedianMSE

mse_metric = MedianMSE(forecast_type="0.5")
```

This module provides the evaluation infrastructure needed for comprehensive benchmarking and comparison of time series forecasting models.
