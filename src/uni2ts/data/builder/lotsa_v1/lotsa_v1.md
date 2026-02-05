# LOTSA V1 Data Builders

The `lotsa_v1` module contains builders for the LOTSA (Large-scale Open Time Series Archive) V1 dataset collection. These builders are responsible for downloading, preprocessing, and saving various time series datasets into a unified Hugging Face `datasets` format.

## Files

- [`_base.py`](_base.py): Defines the base class for LOTSA dataset builders.
- [`gluonts.py`](gluonts.py): Builder for datasets from the GluonTS repository (including Monash datasets).
- [`buildings_bench.py`](buildings_bench.py): Builder for the BuildingsBench dataset.
- [`cloudops_tsf.py`](cloudops_tsf.py): Builder for CloudOps TSF data.
- [`cmip6.py`](cmip6.py): Builder for CMIP6 climate data.
- [`era5.py`](era5.py): Builder for ERA5 reanalysis data.
- [`largest.py`](largest.py): Builder for the LARGEST dataset.
- [`lib_city.py`](lib_city.py): Builder for LibCity traffic data.
- [`proenfo.py`](proenfo.py): Builder for ProEnfo energy data.
- [`subseasonal.py`](subseasonal.py): Builder for subseasonal climate data.
- [`others.py`](others.py): Builder for other various datasets.
- [`__main__.py`](__main__.py): Entry point for building all datasets in LOTSA V1.

## Key Classes

### `LOTSADatasetBuilder` (in `_base.py`)

An abstract base class inheriting from `DatasetBuilder`.

#### Purpose
Provides a common interface and shared logic for all LOTSA V1 dataset builders, which are backed by Hugging Face datasets.

#### Methods
- `load_dataset(transform_map)`: Loads the datasets specified in the builder's `dataset_list`. It uses `HuggingFaceDatasetIndexer` for fast access and applies transformations from the provided `transform_map`.
- `_get_transform(transform_map, dataset)`: Logic for resolving which transformation to apply to a specific dataset.

### `GluonTSDatasetBuilder` (in `gluonts.py`)

#### Purpose
Downloads and processes datasets available in the GluonTS repository, converting them into the LOTSA Hugging Face format.

#### Methods
- `build_dataset(dataset)`: Main method to download and convert a specific GluonTS dataset. It handles metadata extraction, frequency normalization, and filtering (e.g., removing very short series).

## Inter-dependencies
- **Hugging Face `datasets`**: Used as the underlying storage format for preprocessed data.
- **GluonTS**: Used for downloading source datasets and basic metadata handling.
- **`uni2ts.data.indexer`**: Uses `HuggingFaceDatasetIndexer` for efficient data access during loading.
- **`uni2ts.data.dataset`**: Uses `TimeSeriesDataset` and `MultiSampleTimeSeriesDataset` to wrap the indexed data.
- **`uni2ts.common.env`**: Retrieves storage paths from environment variables.

## Connection Flow
1. **Build Phase**: Running `python -m uni2ts.data.builder.lotsa_v1` executes builders which download raw data, apply initial preprocessing, and save them as Hugging Face datasets on disk (at `env.LOTSA_V1_PATH`).
2. **Loading Phase**: When a training or evaluation script runs, it instantiates a builder (e.g., `GluonTSDatasetBuilder`) with a list of datasets.
3. **Indexing**: The builder's `load_dataset` method creates a `HuggingFaceDatasetIndexer` for each dataset on disk.
4. **Wrapping**: The indexers are wrapped in `TimeSeriesDataset` or `MultiSampleTimeSeriesDataset` objects, which are then combined (e.g., via `ConcatDataset`) and returned for use by the `DataLoader`.
