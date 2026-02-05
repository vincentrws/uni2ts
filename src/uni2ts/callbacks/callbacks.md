# Callbacks Module

The `callbacks` module contains custom PyTorch Lightning callbacks used during the training process of MOIRAI models.

## Files

- [`HuggingFaceCheckpoint.py`](HuggingFaceCheckpoint.py): Implements a callback to save models in the Hugging Face `pretrained` format.

## Key Classes

### `HuggingFaceCheckpoint`

Inherits from `lightning.pytorch.callbacks.ModelCheckpoint`.

#### Purpose
To ensure that models are saved in a format compatible with the Hugging Face Hub, allowing for easy sharing and loading using the `from_pretrained` method.

#### Methods
- `_save_checkpoint(trainer, filepath)`: Overrides the default saving logic. It extracts the core model from the Lightning module and calls `save_pretrained(save_dir)`. It also handles removing the `.ckpt` extension from the filepath to create a directory for the Hugging Face format.
- `_remove_checkpoint(trainer, filepath)`: Handles the removal of the saved model directory when a checkpoint is deleted (e.g., if it's no longer one of the top-k checkpoints).

## Inter-dependencies
- **PyTorch Lightning**: Extends the standard `ModelCheckpoint` functionality.
- **Hugging Face Hub**: Relies on the model's `save_pretrained` method (typically provided by `PyTorchModelHubMixin`).
- **`uni2ts.model`**: Interacts with the model instances held within the `LightningModule`.

## Connection Flow
1. **Trainer Loop**: During training, the Lightning `Trainer` triggers callback hooks at specified intervals (e.g., end of epoch).
2. **Checkpoint Trigger**: `HuggingFaceCheckpoint` decides if a checkpoint should be saved based on the monitored metric.
3. **Save Execution**: If saving is triggered, `_save_checkpoint` is called, which in turn calls the model's `save_pretrained` method, resulting in a directory containing `model.safetensors` (or `pytorch_model.bin`) and configuration files.
