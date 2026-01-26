a
# MOIRAI System Patterns

## System Architecture Overview

MOIRAI follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface Layer                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │    CLI       │  │   Python     │  │   Jupyter   │ │
│  │   Tools      │  │     API      │  │  Notebooks  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  Configuration Layer                    │
│              Hydra YAML + Environment Variables         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Training   │  │  Evaluation  │  │  Inference   │ │
│  │   Pipeline   │  │   Pipeline   │  │   Pipeline   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    Model Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Moirai     │  │  Moirai-MoE  │  │   Moirai-2   │ │
│  │   Models     │  │   Models     │  │   Models     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    Core Components                      │
│  ┌────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐    │
│  │  Data  │ │ Transform│ │  Loss   │ │   Eval   │    │
│  │ Loading│ │   Engine │ │ Functions│ │ Metrics  │    │
│  └────────┘ └──────────┘ └─────────┘ └──────────┘    │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                Infrastructure Layer                     │
│   PyTorch │ Lightning │ GluonTS │ HuggingFace │ CUDA   │
└─────────────────────────────────────────────────────────┘
```

## Core Design Patterns

### 1. Modular Component Design

**Pattern**: Each major functionality is encapsulated in its own module with clear interfaces.

**Implementation**:
```
src/uni2ts/
├── model/        # Model architectures
├── data/         # Data loading and transformations
├── module/       # Reusable neural network components
├── distribution/ # Probabilistic output distributions
├── loss/         # Loss functions
├── optim/        # Optimizers and schedulers
└── eval_util/    # Evaluation utilities
```

**Benefits**:
- Easy to test individual components
- Components can be swapped without affecting others
- Clear separation of concerns

### 2. Transformation Chain Pattern

**Pattern**: Data preprocessing uses a fluent interface for chaining transformations.

**Implementation**:
```python
class Transformation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        ...
    
    def chain(self, other: "Transformation") -> "Chain":
        return Chain([self, other])

# Usage:
transform = (
    AddTimeIndex()
    .chain(AddObservedMask())
    .chain(Patch(patch_size=32))
    .chain(Pack(scaler="std"))
)
```

**Benefits**:
- Declarative data pipeline definition
- Easy to add/remove transformations
- Reproducible preprocessing

### 3. Abstract Base Classes (ABC)

**Pattern**: All major components define abstract base classes for extensibility.

**Implementation**:
```python
class DistributionOutput(abc.ABC):
    @abc.abstractmethod
    def distribution(self, distr_args, **kwargs) -> Distribution:
        ...

class Transformation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        ...

class Loss(abc.ABC):
    @abc.abstractmethod
    def forward(self, ...) -> torch.Tensor:
        ...
```

**Benefits**:
- Clear contracts for implementations
- Easy to add new distributions/transforms
- Type checking with mypy

### 4. Builder Pattern for Models

**Pattern**: Models are constructed using builder classes that handle configuration.

**Implementation**:
```python
class MoiraiForecast:
    def __init__(
        self,
        module: MoiraiModule,
        prediction_length: int,
        context_length: int,
        patch_size: str | int,
        ...
    ):
        ...

    def create_predictor(self, batch_size: int) -> Predictor:
        ...

# Usage:
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
    prediction_length=96,
    context_length=512,
    patch_size=32,
    ...
)
predictor = model.create_predictor(batch_size=32)
```

**Benefits**:
- Encapsulates complex construction logic
- Separates model definition from predictor creation
- Easy to test and maintain

## Key Technical Decisions

### 1. Multi-Patch Size Projection

**Decision**: Use multiple projection layers for different patch sizes instead of single fixed patch size.

**Rationale**:
- Different frequencies require different temporal granularities
- Large patches reduce compute for high-frequency data
- Small patches preserve detail for low-frequency data

**Trade-offs**:
- (+) Handles all frequencies in single model
- (+) Optimizes compute/memory usage
- (-) Increases model parameters slightly
- (-) Requires frequency detection logic

### 2. Binary Attention Bias for Variates

**Decision**: Use binary attention bias instead of learned variate embeddings.

**Rationale**:
- Scales to unlimited number of variates
- Equivariant to variable ordering
- Simpler and more efficient than embeddings

**Trade-offs**:
- (+) No limit on variate count
- (+) Order-invariant
- (+) Fewer parameters
- (-) Less expressiveness than learned embeddings
- (-) May need fine-tuning for very high-dimensional data

### 3. Mixture Distribution Output

**Decision**: Output mixture of four parametric distributions instead of single distribution.

**Rationale**:
- Different data types have different distributional properties
- Handles count data, skewed data, and normal data
- Provides flexibility across diverse domains

**Trade-offs**:
- (+) Covers diverse data characteristics
- (+) Automatic distribution selection
- (-) More complex inference
- (-) Slightly slower prediction

### 4. Sequence Packing for Training

**Decision**: Pack multiple short sequences into single training example.

**Rationale**:
- Reduces padding waste from ~61% to <0.4%
- Enables large-batch training on limited memory
- Dramatically improves training efficiency

**Trade-offs**:
- (+) Efficient memory usage
- (+) Faster training
- (+) Better utilization of GPU
- (-) More complex data loading logic
- (-) Requires careful masking

### 5. Masked Encoder Training Objective

**Decision**: Use BERT-style masked prediction instead of autoregressive forecasting.

**Rationale**:
- Learns bidirectional representations
- Better for zero-shot generalization
- More efficient training than autoregressive

**Trade-offs**:
- (+) Better zero-shot performance
- (+) Parallelizable training
- (-) May require more data
- (-) Less natural for sequential generation

### 6. Multi-Architecture Variant Design

**Decision**: Support three different MOIRAI architectures optimized for different trade-offs.

**Moirai-1.0/1.1-R**: Full probabilistic forecasting
- Multi-patch size projections for frequency adaptation
- Transformer with binary attention bias and rotary embeddings
- Mixture distribution output providing full uncertainty quantification
- Complete training pipeline (pretrain, finetune, forecast)

**Moirai-MoE-1.0-R**: Efficient capacity scaling
- Mixture-of-Experts routing for conditional computation
- Enhanced feature processing with residual projections
- Autocorrective inference with causal attention masking
- Expert specialization for improved efficiency/cost ratio

**Moirai-2.0-R**: Deterministic and efficient
- Single patch size with simplified projections
- Direct quantile regression without distribution sampling
- Recursive forecasting for long horizons
- Optimized for production deployment and low latency

**Rationale**:
- Single architecture cannot serve all use cases effectively
- Different applications prioritize speed vs. accuracy vs. uncertainty
- Provides flexibility for researchers and practitioners

**Trade-offs**:
- (+) Serves diverse deployment needs
- (+) Allows optimization for specific scenarios
- (+) Provides migration paths and feature experiments
- (-) Increases complexity and maintenance burden
- (-) Users must select appropriate variant for use case

### 7. Data Pipeline Layering

**Builder Pattern**: Dataset construction abstractions
```python
# Dataset creation (CSV → HuggingFace)
builder = SimpleDatasetBuilder(dataset="mydata")
builder.build_dataset("data.csv", dataset_type="wide")

# Multi-dataset combination
combined = ConcatDatasetBuilder([builder1, builder2])
dataset = combined.load_dataset(transform_map)
```

**Indexer Pattern**: Random access and efficient retrieval
```python
# HuggingFace dataset indexing
indexer = HuggingFaceDatasetIndexer(dataset, uniform=True)

# Fast sequential/random access
batch = indexer[0:32]  # Slicing support
batch = indexer[[5, 12, 25]]  # Arbitrary indices
```

**Transform Chain**: Declarative data preprocessing
```python
# Composable transformation pipeline
transform = (
    AddTimeIndex()
    .chain(Patch(patch_size=32))
    .chain(Pack(scaler="std"))
)
```

**Rationale**:
- Separation of concerns between data ingestion, access, and processing
- Streamlined pipeline enables reproducibility and debugging
- Pattern enables extensibility for new data formats and models

### 8. Attention Mechanism Variations

**Binary Attention Bias**: Same/different variate relationships
- Learned attention masks between time series variables
- Enables cross-variate information flow
- Scalable to high-dimensional multivariate data

**Causal Attention**: Temporal coherence for generation
- Prevents future information leakage
- Enables autoregressive and autocorrective patterns
- Critical for MoE's iterative refinement inference

**Rotary Position Embeddings**: Time-aware positioning
- Relative temporal distances in attention
- Better handling of variable frequencies
- Improved long-range dependencies

### 6. Collective OHLC Normalization (StockMarket-MOIRAI)

**Decision**: For financial OHLCV data, normalize Open+High+Low+Close collectively while keeping Volume separate.

**Rationale**:
- OHLC represent the same underlying asset price during a time period
- Collective normalization preserves relative relationships between price components
- Volume has different distribution and scale than prices

**Implementation**:
```python
class CollectiveOHLCScaler(PackedScaler):
    """
    Normalize OHLC collectively (shared mean/std) but Volume separately.
    
    Args:
        ohlc_indices: List of indices for OHLC columns [0,1,2,3]
        volume_indices: List of indices for Volume columns [4]
    """
    def fit(self, data):
        # Compute single mean/std across all OHLC variates
        ohlc_data = data[..., self.ohlc_indices]  # [..., 4]
        self.ohlc_mean = ohlc_data.mean(dim=(-2, -1), keepdim=True)
        self.ohlc_std = ohlc_data.std(dim=(-2, -1), keepdim=True)
        
        # Volume normalization (IndividualStdScaler behavior)
        volume_data = data[..., self.volume_indices]
        self.vol_mean = volume_data.mean(dim=-2, keepdim=True)
        self.vol_std = volume_data.std(dim=-2, keepdim=True)
    
    def transform(self, data):
        # Apply collective transform to OHLC
        data[..., self.ohlc_indices] = (
            data[..., self.ohlc_indices] - self.ohlc_mean
        ) / self.ohlc_std
        
        # Apply individual transform to Volume
        data[..., self.volume_indices] = (
            data[..., self.volume_indices] - self.vol_mean
        ) / self.vol_std
        
        return data
```

### 7. Semantic Attention Bias (StockMarket-MOIRAI)

**Decision**: Extend binary attention from "same/different variate" to semantic type relationships for OHLCV.

**Rationale**:
- OHLC have different semantic roles (Close is target, Open is reference)
- Volume provides context but different from price information
- Learned type relationships can capture financial domain knowledge

**Implementation**:
```python
class SemanticAttentionBias(nn.Module):
    """
    Learn attention patterns between OHLCV semantic types.
    
    Type IDs:
    - 0: Close (target price, most important)
    - 1: Open (reference price)
    - 2: High (range upper bound)
    - 3: Low (range lower bound)
    - 4: Volume (trading volume)
    """
    
    def __init__(self, num_types=5, hidden_dim=64):
        super().__init__()
        self.type_embeddings = nn.Embedding(num_types, hidden_dim)
        self.bias_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output single bias value
        )
    
    def forward(self, query_ids, key_ids):
        """
        Args:
            query_ids: [batch, seq_len] - semantic type IDs
            key_ids: [batch, seq_len] - semantic type IDs
        """
        query_emb = self.type_embeddings(query_ids)  # [batch, seq_len, hidden]
        key_emb = self.type_embeddings(key_ids)      # [batch, seq_len, hidden]
        
        # Compute pairwise type relationships
        bias_matrix = torch.zeros(batch, seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                pair_emb = torch.cat([query_emb[:, i], key_emb[:, j]], dim=-1)
                bias_matrix[:, i, j] = self.bias_network(pair_emb).squeeze(-1)
        
        return bias_matrix
```

**Benefits**:
- Model learns that Close↔Close interactions are important
- Open provides reference context for Close predictions
- Volume gets appropriate attention weight relative to prices

## Component Relationships

### Model Architecture

```
MoiraiModule (Base)
├── Input Projection
│   ├── MultiInSizeLinear (multiple patch sizes)
│   └── Position Encoding (RoPE)
├── Transformer Encoder
│   ├── Multi-Head Attention
│   │   ├── Binary Attention Bias (variates)
│   │   └── Rotary Position Embeddings (time)
│   ├── Feed-Forward Networks
│   └── Layer Normalization
└── Output Projection
    └── MixtureDistributionOutput
        ├── StudentT
        ├── NegativeBinomial
        ├── LogNormal
        └── Normal
```

### Data Pipeline

```
Raw Data
    ↓
GluonTS Dataset
    ↓
Uni2TS Transformations
    ├── AddTimeIndex
    ├── AddObservedMask
    ├── Patch
    ├── Pack (sequence packing)
    └── AddPredictionMask
    ↓
DataLoader
    ├── MultiSampleTimeSeriesDataset
    ├── PackCollate (batching)
    └── Prefetching
    ↓
Packed Batches
```

### Training Loop

```
Lightning Trainer
    ├── Model: MoiraiModule
    ├── DataLoader: Packed batches
    ├── Loss: PackedNLLLoss
    ├── Optimizer: AdamW
    └── Scheduler: CosineAnnealingLR
```

## Critical Implementation Paths

### 1. Zero-Shot Inference

**Path**:
```python
# 1. Load pre-trained model
model = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")

# 2. Create predictor
forecast = MoiraiForecast(
    module=model,
    prediction_length=96,
    context_length=512,
    ...
)
predictor = forecast.create_predictor(batch_size=32)

# 3. Load data (GluonTS)
ds = PandasDataset(dict(df))

# 4. Split and generate windows
train, test_template = split(ds, offset=-100)
test_data = test_template.generate_instances(
    prediction_length=96,
    windows=10,
    distance=96
)

# 5. Predict
forecasts = predictor.predict(test_data.input)
```

### 2. Fine-Tuning

**Path**:
```python
# 1. Load pre-trained model
model = MoiraiModule.load_from_checkpoint(
    "Salesforce/moirai-1.1-R-small"
)

# 2. Create LightningModule
lit_model = MoiraiFinetuneModule(
    model=model,
    loss_fn=PackedNLLLoss()
)

# 3. Setup data
train_data = FinetuneDataset(
    indexer=HFDatasetIndexer(...),
    transform=transform_chain,
    offset=-100
)

# 4. Train
trainer = LightningTrainer(
    max_epochs=10,
    accelerator="gpu"
)
trainer.fit(lit_model, train_data)
```

### 3. Evaluation

**Path**:
```python
# 1. Create predictor
predictor = model.create_predictor(batch_size=32)

# 2. Load test data
test_data = EvalDataset(
    indexer=...,
    transform=...,
    windows=100
)

# 3. Generate forecasts
forecasts = predictor.predict(test_data.input)

# 4. Compute metrics
metrics = evaluate_forecasts(
    forecasts=test_data.label,
    targets=test_data.output,
    metrics=["MSE", "MASE", "CRPS"]
)
```

## Design Patterns Summary

| Pattern | Location | Purpose |
|---------|----------|---------|
| Module Pattern | `src/uni2ts/module/` | Reusable neural network components |
| Strategy Pattern | `src/uni2ts/distribution/` | Interchangeable distributions |
| Builder Pattern | `src/uni2ts/model/` | Complex model construction |
| Chain of Responsibility | `src/uni2ts/transform/` | Data transformation pipeline |
| Template Method | `src/uni2ts/loss/` | Base loss implementations |
| Factory Pattern | `src/uni2ts/data/builder/` | Dataset creation |

## Anti-Patterns to Avoid

1. **Direct instantiation of models**: Always use `from_pretrained()` for consistency
2. **Hardcoded transformations**: Use transformation chains for reproducibility
3. **Manual tensor manipulation**: Use packed sequences API for efficiency
4. **Bypassing type hints**: Leverage jaxtyping for tensor shape safety
5. **Ignoring masking**: Always handle observed masks for proper training