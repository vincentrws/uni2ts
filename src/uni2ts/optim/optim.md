# Optimization Module

The `optim` module provides utilities for optimizing MOIRAI models, primarily focusing on learning rate scheduling. It offers a unified interface for various scheduling strategies common in large-scale transformer training.

## Files

- [`lr_scheduler.py`](lr_scheduler.py): Implements several learning rate schedules with warmup and restart capabilities.

## Key Functions & Classes

### `get_scheduler` (in `lr_scheduler.py`)

#### Purpose
A unified factory function that returns a PyTorch learning rate scheduler based on a name or `SchedulerType` enum.

### `SchedulerType` (in `lr_scheduler.py`)

#### Supported Schedules
- `LINEAR`: Linear decay after warmup.
- `COSINE`: Cosine decay after warmup.
- `COSINE_WITH_RESTARTS`: Cosine decay with hard restarts.
- `POLYNOMIAL`: Polynomial decay after warmup.
- `CONSTANT`: Fixed learning rate.
- `CONSTANT_WITH_WARMUP`: Linear warmup followed by a constant rate.
- `INVERSE_SQRT`: Inverse square root decay (common in original Transformer training).
- `REDUCE_ON_PLATEAU`: Standard PyTorch plateau reduction.

### Individual Schedule Functions
Functions like `get_cosine_schedule_with_warmup` implement the logic for each type, often using `torch.optim.lr_scheduler.LambdaLR` to define custom decay curves.

## Inter-dependencies
- **PyTorch**: Extends standard `torch.optim.lr_scheduler` classes.
- **`uni2ts.model`**: Used by `MoiraiPretrain` and `MoiraiFinetune` in their `configure_optimizers` method.

## Connection Flow
1.  **Configuration**: The desired scheduler and its parameters (e.g., `num_warmup_steps`) are specified in the Hydra configuration.
2.  **Instantiation**: During `configure_optimizers`, the model calls `get_scheduler` with the optimizer and the configured parameters.
3.  **Step Execution**: PyTorch Lightning automatically calls the scheduler's `step()` method at the specified interval (typically every training step), updating the optimizer's learning rate according to the curve.
