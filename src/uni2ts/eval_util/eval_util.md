# Evaluation Utilities Module

The `eval_util` module provides tools for evaluating forecasting models. It leverages GluonTS's evaluation framework and extends it with custom metrics and support for varied dataset formats.

## Files

- [`evaluation.py`](evaluation.py): Core evaluation logic, including model prediction and metric computation.
- [`metrics.py`](metrics.py): Definitions for additional evaluation metrics (e.g., MedianMSE).
- [`data.py`](data.py): (Not read, but likely handles data loading for evaluation)
- [`plot.py`](plot.py): (Not read, but likely contains plotting utilities for forecasts)
- [`_hf_dataset.py`](_hf_dataset.py), [`_lsf_dataset.py`](_lsf_dataset.py), [`_pf_dataset.py`](_pf_dataset.py): Dataset-specific evaluation helpers.

## Key Functions & Classes

### `evaluate_model` (in `evaluation.py`)

#### Purpose
The primary entry point for evaluating a `Predictor` on a `TestData` set.

#### Process
1.  Calls `model.predict()` on the test data inputs to generate `forecasts`.
2.  Passes the forecasts to `evaluate_forecasts` to compute metrics.

### `evaluate_forecasts` (in `evaluation.py`)

#### Purpose
Computes metrics by comparing generated `forecasts` with true labels from `test_data`.

#### Features
- **Batch Processing**: Processes forecasts in batches to manage memory.
- **Axis Aggregation**: Supports aggregating metrics across different dimensions:
    - `None`: Overall aggregate.
    - `0`: Per-dataset item aggregation.
    - `1` or `2`: Temporal dimension aggregation.
- **Seasonal Error**: Automatically computes seasonal error for scaled metrics like MASE.

### `BatchForecast` (in `evaluation.py`)

#### Purpose
A wrapper for a list of GluonTS `Forecast` objects that adds a batch dimension. This ensures compatibility with `gluonts.ev` (GluonTS Evaluation) metrics which expect batched inputs.

### `MedianMSE` (in `metrics.py`)

#### Purpose
Computes the Mean Squared Error using the median (0.5 quantile) of the forecast distribution.

## Inter-dependencies
- **GluonTS**: Heavily relies on `gluonts.model`, `gluonts.ev`, and `gluonts.dataset`.
- **Pandas**: Returns evaluation results as a `pd.DataFrame`.
- **NumPy**: Used for tensor operations and handling masked arrays.
- **Tqdm**: Provides progress bars for the evaluation loop.

## Connection Flow
1.  **Prediction**: The model (a `Predictor`) takes test inputs and generates probabilistic forecasts.
2.  **Batching**: Forecasts and labels are grouped into batches.
3.  **Data Preparation**: `_get_data_batch` combines forecasts, labels, and seasonal information into a `ChainMap`.
4.  **Metric Update**: Each metric evaluator is updated with the current batch of data.
5.  **Aggregation**: Once all batches are processed, `evaluator.get()` computes the final aggregated metrics.
6.  **Reporting**: Results are flattened into a Pandas DataFrame for analysis and logging.
