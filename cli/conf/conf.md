# Configuration Module (Hydra)

The `cli/conf` directory contains the Hydra configuration files that define how models are pretrained, fine-tuned, and evaluated. It uses a modular structure where different aspects of an experiment (model, data, trainer) are defined in separate YAML files and composed using Hydra's `defaults` system.

## Directory Structure

- [`pretrain/`](pretrain/): Configurations for the pretraining stage.
- [`finetune/`](finetune/): Configurations for dataset-specific fine-tuning.
- [`eval/`](eval/): Configurations for running benchmarks and evaluations.

## Key Configuration Files

### `default.yaml` (in each sub-directory)
The base configuration for that stage. It defines the `trainer` (PyTorch Lightning), `train_dataloader`, and general settings like `seed`, `precision`, and `max_epochs`.

### `model/*.yaml`
Defines the model architecture and its hyperparameters.
- Uses `_target_` to specify the LightningModule class (e.g., `uni2ts.model.moirai.MoiraiPretrain`).
- Configures `module_kwargs` for the underlying Transformer.
- Sets optimization parameters like `lr`, `weight_decay`, and `num_warmup_steps`.

### `data/*.yaml`
Defines the dataset(s) to be used.
- For pretraining: Points to LOTSA V1 unweighted or weighted collections.
- For fine-tuning/eval: Points to specific datasets like ETTh1, Monash, or GluonTS benchmarks.

## Hydra Features Used

### Custom Resolvers (from `uni2ts.common.hydra_util`)
- `as_tuple`: Used for `patch_sizes` which must be a tuple in Python.
- `cls_getattr`: Used to dynamically retrieve static attributes from model classes (e.g., `seq_fields`).
- `floordiv` & `mul`: Used for calculating steps and intervals relative to other parameters.

### Interpolation
Configs frequently use `${...}` syntax to reference other parts of the config, ensuring consistency (e.g., setting `max_length` in the collator based on the model's `max_seq_len`).

### Composition
Users can override any part of the config from the command line:
`python -m cli.train model=moirai_small data=proenfo trainer.precision=16`

## Inter-dependencies
- **`uni2ts.model`**: Configs provide the parameters for instantiating these classes.
- **`uni2ts.data`**: Configs define which builders and loaders to use.
- **PyTorch Lightning**: The `trainer` section directly configures the Lightning `Trainer`.

## Connection Flow
1.  **Selection**: The user selects a base config (e.g., `pretrain/default`).
2.  **Composition**: Hydra merges the selected model and data YAMLs into the base config.
3.  **Resolution**: Custom resolvers are executed to compute final values.
4.  **Instantiation**: The CLI script calls `hydra.utils.instantiate(cfg.model)` and `hydra.utils.instantiate(cfg.trainer)`, etc., to create the actual Python objects.
5.  **Execution**: The instantiated objects are passed to Lightning for the training or evaluation loop.
