# CLI Module

The `cli` module provides the command-line interface for training, fine-tuning, and evaluating MOIRAI models. It uses **Hydra** for configuration management and **PyTorch Lightning** for distributed training and evaluation orchestration.

## Files

- [`train.py`](train.py): Main script for model pretraining and fine-tuning.
- [`eval.py`](eval.py): Main script for evaluating models on benchmark datasets.
- [`conf/`](conf/conf.md): Directory containing Hydra YAML configuration files.

## Key Components

### `train.py`

#### Purpose
Orchestrates the training process using PyTorch Lightning's `Trainer`.

#### Logic
1.  **Hydra Configuration**: Loads the experiment configuration from `cli/conf/`.
2.  **Model Instantiation**: Creates the `LightningModule` (e.g., `MoiraiPretrain` or `MoiraiFinetune`).
3.  **Data Loading**: Instantiates the `DatasetBuilder` and loads the datasets, applying the model's specified transformations.
4.  **Lightning Integration**: Wraps the datasets in a `DataModule` which handles distributed sampling and collation (packing).
5.  **Execution**: Calls `trainer.fit()` to start the training/fine-tuning loop.

### `eval.py`

#### Purpose
Executes model evaluation and computes performance metrics.

#### Logic
1.  **Data Setup**: Loads test data and metadata from the specified benchmark.
2.  **Model Setup**: Instantiates the model (wrapping it as a `Predictor`) with hyperparameters matching the test data (e.g., matching `prediction_length`).
3.  **Evaluation**: Calls `evaluate_model` from `uni2ts.eval_util`, which generates forecasts and computes metrics.
4.  **Logging**: Prints results and logs metrics to TensorBoard.
5.  **OOM Handling**: Includes a loop that automatically reduces the batch size if a `torch.cuda.OutOfMemoryError` is encountered.

## Inter-dependencies
- **`uni2ts.model`**: CLI scripts instantiate and wrap these models.
- **`uni2ts.data`**: CLI scripts use builders and custom data loaders.
- **`uni2ts.eval_util`**: `eval.py` relies on this for metric computation.
- **Hydra/OmegaConf**: Used for all configuration.
- **PyTorch Lightning**: The underlying execution engine.

## Connection Flow
1.  **Command Entry**: The user runs a command like `python -m cli.train data=lotsa_v1_weighted model=moirai_base`.
2.  **Config Resolution**: Hydra resolves the YAML files into a single `DictConfig`.
3.  **Component Creation**: The script uses `hydra.utils.instantiate` to create the model, trainer, and data builder.
4.  **Data Pipeline**: The builder creates the dataset, the dataset applies transforms, and the loader creates packed batches.
5.  **Execution Loop**: Lightning manages the training steps, logging, and checkpointing (using the `HuggingFaceCheckpoint` callback).
6.  **Reporting**: Final models are saved, and metrics are logged.
