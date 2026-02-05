# MOIRAI Model Module

The `moirai` module contains the implementation of the Masked Encoder-based Universal Time Series Forecasting Transformer (MOIRAI). It includes the core neural network architecture and specialized classes for pretraining, fine-tuning, and inference.

## Files

- [`module.py`](module.py): Defines the core `MoiraiModule` (the Transformer architecture).
- [`pretrain.py`](pretrain.py): Implements `MoiraiPretrain`, a LightningModule for large-scale pretraining with masked prediction.
- [`finetune.py`](finetune.py): Implements `MoiraiFinetune`, a LightningModule for task-specific fine-tuning.
- [`forecast.py`](forecast.py): Implements `MoiraiForecast`, a LightningModule wrapper for inference and GluonTS integration.

## Key Classes

### `MoiraiModule` (in `module.py`)

The heart of the system, inheriting from `nn.Module` and `PyTorchModelHubMixin`.

#### Architecture
- **Input Projections**: `MultiInSizeLinear` handles variable patch sizes by having multiple linear layers.
- **Encoder**: `TransformerEncoder` with specialized attention:
    - **Rotary Position Embeddings (RoPE)**: For temporal encoding.
    - **Binary Attention Bias**: For variate-specific attention masking.
- **Scaling**: `PackedStdScaler` for data standardization.
- **Output**: `DistrParamProj` maps hidden states to distribution parameters.

### `MoiraiPretrain` (in `pretrain.py`)

#### Purpose
Used for training the model on the LOTSA dataset using a masked modeling objective.

#### Features
- **Transform Map**: Defines a complex chain of transforms for pretraining, including random dimension sampling, patching, and random masking.
- **Loss**: Defaults to `PackedNLLLoss`.
- **Optimizer**: Configures AdamW with weight decay exclusion for normalization and bias parameters.

### `MoiraiFinetune` (in `finetune.py`)

#### Purpose
Used for adapting a pretrained MOIRAI model to a specific dataset or forecasting task.

#### Features
- **Trainable Parameters**: Supports patterns like "full" finetuning, freezing FFN layers, or "head-only" adaptation.
- **Specialized Transforms**: Uses `EvalCrop` and `EvalPad` to handle specific context/prediction lengths.

### `MoiraiForecast` (in `forecast.py`)

#### Purpose
The inference wrapper.

#### Features
- **Auto-Patch Selection**: Automatically selects the best patch size for a given time series by evaluating NLL on a validation window within the context.
- **GluonTS Integration**: Creates a `PyTorchPredictor` for seamless use with GluonTS evaluation pipelines.

## Inter-dependencies
- **`uni2ts.module`**: Uses transformer, attention, and embedding components.
- **`uni2ts.distribution`**: Uses `DistributionOutput` for probabilistic heads.
- **`uni2ts.loss.packed`**: Uses specialized losses for training on packed sequences.
- **`uni2ts.transform`**: Relies heavily on the transformation pipeline for data preparation.
- **PyTorch Lightning**: All training/finetuning classes are `LightningModule`s.

## Connection Flow
1.  **Initialization**: A `MoiraiModule` is created with a specific backbone size (base, large, etc.).
2.  **Orchestration**: `MoiraiPretrain` or `MoiraiFinetune` wraps the module for training.
3.  **Data Flow**: `train_transform_map` prepares raw datasets into packed batches.
4.  **Forward Pass**: `MoiraiModule.forward` standardizes data, projects patches, applies the Transformer, and outputs a distribution.
5.  **Backward Pass**: `PackedLoss` computes the gradient while respecting sequence packing.
6.  **Inference**: `MoiraiForecast` loads the pretrained module, selects a patch size, and generates forecast samples.
