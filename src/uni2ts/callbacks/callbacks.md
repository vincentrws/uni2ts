# Callbacks Module

## Overview

The callbacks module provides PyTorch Lightning training callbacks for the MOIRAI time series forecasting models. It currently focuses on checkpoint management and model persistence.

## Components

### HuggingFaceCheckpoint

A specialized PyTorch Lightning callback that extends the base `ModelCheckpoint` class to save models in Hugging Face format.

**Key Features:**
- Inherits from `pl.callbacks.ModelCheckpoint`
- Saves models using the Hugging Face `save_pretrained()` method
- Handles both distributed and single-process training
- Automatic checkpoint cleanup when saving new checkpoints

**Constructor Parameters:**
- `dirpath`: Directory path for saving checkpoints (optional)
- `filename`: Checkpoint filename template (optional)
- `monitor`: Metric to monitor for saving (optional)
- `save_top_k`: Number of top checkpoints to keep
- `mode`: Min/max mode for metric monitoring
- `every_n_epochs`: Save frequency in epochs

**Core Methods:**
- `_save_checkpoint()`: Overrides Lightning's save method to extract the MOIRAI model from the LightningModule wrapper and save it in Hugging Face format to a directory derived from the checkpoint filepath
- `_remove_checkpoint()`: Removes old checkpoint directories when cleanup is needed

**Usage Pattern:**
```python
from src.uni2ts.callbacks import HuggingFaceCheckpoint

checkpoint_callback = HuggingFaceCheckpoint(
    dirpath="checkpoints/",
    monitor="val_loss",
    save_top_k=3,
    mode="min",
    every_n_epochs=5
)

trainer = pl.Trainer(callbacks=[checkpoint_callback])
```

**Integration:**
- Designed specifically for MOIRAI models that have a `save_pretrained()` method
- Safely handles nested module structures in Lightning modules
- Provides warnings for model extraction issues
- Integrates with Lightning's logging system

This callback enables seamless model persistence in the Hugging Face ecosystem, allowing trained MOIRAI models to be easily shared, versioned, and deployed.