# MOIRAI Active Context

## Current Work Focus

As of January 2026, the StockMarket-MOIRAI project is focused on adapting the universal MOIRAI foundation model for specialized OHLCV stock market price prediction.

### Active Projects

#### 1. OHLCV Normalization (Collective Scaling)
- **Status**: Completed (February 2026)
- **Problem**: MOIRAI's PackedStdScaler normalizes each variate independently, but financial OHLC prices share the same unit and range, necessitating collective normalization to preserve their relative relationships.
- **Solution**: 
    - Implemented `GroupedPackedStdScaler` for arbitrary variate grouping.
    - Implemented specialized `OHLCVPackedScaler` for financial data.
    - Verified that including **Close** in the OHLC collective group (Group 0) correctly scales all four price components using the same statistics, while Volume (Group 1) and time features (Groups 7, 8) are scaled appropriately.
- **Location**: `src/uni2ts/module/packed_scaler.py`
- **Tests**: `test/module/test_packed_scaler.py` and `notebooks/4_ohlcv_with_close_test.ipynb`.
- **Key Insight**: Grouping OHLC (40 positions in a 10-step window) ensures price consistency across variates.

#### 2. SemanticAttentionBias for OHLCV
- **Status**: Planned (January 2026)
- **Problem**: Binary attention bias (same/different variate) doesn't capture OHLCV semantics
- **Solution**: Implement type IDs (close=1, open=2, high=3, low=4, volume=5) with learned type relationships
- **Benefits**: Model learns that OHLC are related prices vs. Volume being separate context

#### 3. Directional Accuracy Loss Function
- **Status**: In Development (January 2026)
- **Problem**: Standard NLL loss optimizes exact prediction, traders care about direction
- **Solution**: Add directional component to loss function with configurable weight
- **Metrics**: Directional accuracy (>50% baseline), market regime awareness

#### 4. OHLCV Data Pipeline & Model Integration
- **Status**: Completed (February 2026)
- **Challenge**: Seamlessly integrating custom data packing and scaling into the MOIRAI training workflow.
- **Solutions**:
  - **Standard Pipeline Adaptation**: Utilized the standard `FinetuneDataset` and `FinetunePatchCrop` pipeline instead of a custom loader for maximum compatibility with the existing training infrastructure.
  - **Data Builder**: Created `src/uni2ts/data/builder/turn_parquet_csv.py` to prepare CSVs with all 7 variates (OHLCV + Time Features) pre-calculated.
  - **Custom Packing**: `OHLCVPackedScaler` handles the logic of grouping OHLC variates together for collective scaling.
  - **Custom Module**: Created `OHLCVMoiraiModule` (`src/uni2ts/model/moirai/custom_module.py`) to integrate the standard pipeline with `OHLCVPackedScaler`.
- **Key Implementation Details**:
  - **Data Flow**: Parquet → CSV (with time features) → HuggingFace Dataset → FinetuneDataset → FinetunePatchCrop → Batch.
  - **Windowing**: `FinetunePatchCrop` handles sliding window creation dynamically based on `train_length`, `context_length`, and `distance`.
  - **Scaling**: `OHLCVPackedScaler` applies collective normalization to OHLC groups and individual normalization to Volume and Time features.
- **Documentation**: Detailed walkthrough available in `docs/fine_tune_process.md`.
- **Features**:
  - Direct compatibility with PyTorch DataLoaders.
  - Efficient on-the-fly windowing (no massive pre-cropped dataset storage).
  - Preserves `create_hf_dataset` for backward compatibility/caching.
- **Usage**:
  ```python
  # Loader
  loader = OHLCVLoader(
      data_path='...',
      window_size=512,
      stride=512
  )
  
  # Hydra Config
  # _target_: uni2ts.model.moirai.custom_module.OHLCVMoiraiModule
  ```

## Recent Changes

### January 2026
- **Comprehensive Module Documentation** - Created detailed .md files for key modules
  - `builder.md`: Dataset construction and loading patterns (HuggingFace integration, LOTSA support)
  - `indexer.md`: Data indexing abstractions (PyArrow optimization, sequence access patterns)
  - `moirai.md`: Original MOIRAI 1.0/1.1 (multi-patch, mixture distributions, full training pipeline)
  - `moirai_moe.md`: MoE variant with expert routing and autocorrective inference
  - `moirai2.md`: Simplified quantile regression variant with recursive forecasting
- **PackedMidRangeScaler** implemented for custom normalization ranges
  - Added to `src/uni2ts/module/packed_scaler.py`
  - Implements `(x - mid) / range` formula
  - Useful for technical indicators with known value ranges
  - Comprehensive test coverage added to `test/module/test_packed_scaler.py`

### August 2025
- **Moirai-2.0-R-small** released on HuggingFace Hub
- Enhanced fine-tuning module with better hyperparameter tuning
- Added LSF benchmark fine-tuning examples

### October 2024
- **Moirai-MoE** paper published on arXiv
- Released Moirai-MoE-Small and Moirai-MoE-Base models
- Added inference code and notebook examples
- Enhanced documentation for MoE architecture

### June 2024
- **Moirai-1.1-R** model weights released in small, base, and large variants
- Performance improvements over 1.0-R models
- Updated documentation and examples

### March 2024
- **Initial MOIRAI release** (1.0-R models)
- LOTSA dataset public release
- Uni2TS library published
- Comprehensive documentation and examples

## Next Steps

### Immediate Priorities (Q1 2026)

1. **Model Integration of OHLCV Scaler**
   - Integrate `OHLCVPackedScaler` into MOIRAI fine-tuning module
   - Update Hydra configs to use specialized scaler for financial tasks
   - Test end-to-end training loop with OHLCV data samples
   - Document usage patterns for financial data

2. **OHLCV Data Pipeline Enhancements**
   - Implement Parquet file loader with column validation
   - Add corporate action handling (splits, dividends)
   - Create market gap filling strategies
   - Validate pipeline with multiple asset examples

3. **Directional Accuracy Metrics**
   - Implement directional accuracy evaluation function
   - Add market regime-aware metrics
   - Create backtesting integration points
   - Establish baseline performance targets

### Medium-Term Goals (Q2 2026)

1. **SemanticAttentionBias Architecture**
   - Extend attention mechanism for OHLCV type relationships
   - Implement semantic type embeddings
   - Train with weighted attention on price relationships
   - Validate attention pattern learning

2. **Financial Fine-tuning Framework**
   - Develop curriculum learning for OHLCV data
   - Implement directional-aware loss functions
   - Create specialized training configurations
   - Benchmark against existing financial models

3. **Production Integration**
   - Add trading platform API integrations
   - Implement real-time inference pipelines
   - Create streaming OHLCV data adapters
   - Develop risk management overlays

### Long-Term Vision (2027+)

1. **Multi-Asset Portfolio Forecasting**
   - Extend to multi-stock or portfolio-level predictions
   - Cross-asset correlation modeling
   - Market-sector aware attention mechanisms

2. **Advanced Financial Features**
   - Technical indicator integration as additional variates
   - Order book and flow data incorporation
   - Options pricing and volatility surface modeling
   - Alternative data sources integration

3. **Enterprise Deployment**
   - Regulatory compliance frameworks
   - Model explainability for financial decisions
   - High-frequency trading optimizations
   - Cloud-native deployment architectures

## Active Decisions

### Model Selection Guidance

When to use each model variant:

**Moirai-1.1-R**
- Use when: Need probabilistic forecasts with mixture distributions
- Best for: Diverse domains requiring uncertainty quantification
- Trade-off: Slightly slower inference due to mixture sampling

**Moirai-MoE-1.0-R**
- Use when: Need efficiency with large model capacity
- Best for: Complex patterns requiring expert routing
- Trade-off: More complex architecture, harder to debug

**Moirai-2.0-R**
- Use when: Need deterministic forecasts and interpretability
- Best for: Production deployments requiring fast inference
- Trade-off: No probabilistic uncertainty estimates

### Configuration Best Practices

**Context Length Selection**
- Short-term forecasts: 200-500 patches
- Medium-term forecasts: 500-1000 patches
- Long-term forecasts: 1000+ patches (if memory allows)

**Patch Size Guidelines**
- Yearly/quarterly: 8-16
- Monthly: 16-32
- Weekly/daily: 32
- Hourly: 32-64
- Minute/second: 64-128

**Batch Size Recommendations**
- Training: Start with 32, adjust based on GPU memory
- Inference: Use largest batch that fits memory
- Evaluation: Batch size has minimal impact on metrics

### Data Preparation Patterns

**For Custom Datasets**
1. Use `uni2ts.data.builder.simple` for preprocessing
2. Split with explicit date offset for validation
3. Consider normalization for improved convergence
4. Use appropriate mode ('S' for univariate, 'M' for multivariate)

**For Evaluation**
1. Use rolling window evaluation with non-overlapping windows
2. Set prediction_length based on use case
3. Context length should be 3-10x prediction_length
4. Generate multiple windows for robust metrics

## Important Patterns

### Type Safety with jaxtyping

**Pattern**: Always use tensor shape annotations for function signatures

```python
def forward(
    self,
    target: Float[torch.Tensor, "*batch seq_len max_patch"],
    observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
) -> Distribution:
    ...
```

**Benefit**: Catch tensor shape errors at development time, not runtime

### Transformation Chain Composition

**Pattern**: Build transformations declaratively, not procedurally

```python
# Good
transform = (
    AddTimeIndex()
    .chain(AddObservedMask())
    .chain(Patch(patch_size=32))
    .chain(Pack(scaler="std"))
)

# Avoid
transform1 = AddTimeIndex()
transform2 = AddObservedMask()
result = transform2(transform1(data))
```

**Benefit**: Reproducible pipelines, easier debugging, clear data flow

### Model Loading

**Pattern**: Always use `from_pretrained()` for consistency

```python
# Good
model = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")

# Avoid
model = MoiraiModule(...)  # Manual construction
```

**Benefit**: Guaranteed compatibility, proper weight loading

### Virtual Environment Activation

**Pattern**: Always activate venv before running commands

```bash
# Required
source venv/bin/activate
python -m cli.train ...

# Will fail without activation
python -m cli.train ...  # May not find installed packages
```

**Benefit**: Ensures correct environment and dependencies

## Project Insights

### What Works Well

1. **Zero-Shot Performance**
   - Moirai achieves competitive performance without fine-tuning
   - Particularly strong on diverse datasets with varied frequencies
   - Mixture distributions handle diverse data characteristics

2. **Sequence Packing Efficiency**
   - Dramatically reduces padding waste
   - Enables training on limited GPU memory
   - One of the key innovations for large-scale training

3. **Modular Architecture**
   - Easy to extend with new distributions/transforms
   - Clean separation of concerns
   - Comprehensive test coverage maintains quality

4. **Type Safety**
   - jaxtyping catches errors early
   - Improves code readability
   - Better IDE support with type hints

### Common Pitfalls

1. **Forgetting Virtual Environment** 🚨 CRITICAL 🚨
   - **Error**: ModuleNotFoundError, import errors, "uni2ts package not found"
   - **Solution**: Always run `source venv/bin/activate` before any MOIRAI commands
   - **Example**: 
     ```bash
     # Terminal shows: opt/uni2ts$
     source venv/bin/activate  # Now shows: (venv) opt/uni2ts$
     python -m cli.train ...  # Will work
     ```
   - **Tip**: Commands WILL FAIL without activation. Always check for (venv) in prompt.

2. **Incorrect Patch Size**
   - Error: Poor performance, wasted compute
   - Solution: Match patch size to data frequency
   - Tip: Use "auto" patch size when available

3. **Insufficient Context Length**
   - Error: Poor long-term forecasts
   - Solution: Use 3-10x prediction_length for context
   - Tip: Longer context always improves quality (until memory limit)

4. **Ignoring Data Normalization**
   - Error: Slow convergence, poor results
   - Solution: Normalize during data preparation
   - Tip: Use --normalize flag in data builder

5. **Wrong Mode Selection**
   - Error: Univariate data with mode='M' or vice versa
   - Solution: Mode 'S' for univariate, 'M' for multivariate
   - Tip: Check data shape before selecting mode

### Performance Learnings

1. **Batch Size Impact**
   - Training: Larger batches = faster convergence but may need more epochs
   - Inference: Larger batches = higher throughput but higher latency
   - Trade-off: Batch size depends on latency vs. throughput requirements

2. **GPU Utilization**
   - Moirai models are compute-bound, not memory-bound (after packing)
   - Multiple GPUs scale well with Lightning distributed training
   - Mixed precision provides 50% memory reduction with minimal quality impact

3. **Inference Speed**
   - Moirai-2.0-R fastest (deterministic output)
   - Moirai-1.1-R slower due to mixture sampling
   - Moirai-MoE similar to 1.1-R but more efficient per FLOP

4. **Model Size vs. Performance**
   - Small model good for many use cases
   - Base model provides better quality for complex patterns
   - Large model diminishing returns for most tasks

## Current Limitations

### Known Issues

1. **PyTorch Version Constraints**
   - Limited to 2.1.x-2.4.x
   - Cannot use latest PyTorch features
   - Plan: Evaluate PyTorch 2.5+ compatibility

2. **Context Length Memory**
   - Maximum context limited by GPU memory
   - Very long sequences (>2000 patches) require large GPUs
   - Workaround: Use gradient checkpointing or smaller models

3. **CPU Inference**
   - Slower than GPU, especially for large batches
   - No optimized CPU kernels
   - Future: Investigate ONNX export for optimized CPU inference

4. **Real-Time Deployment**
   - No streaming inference support
   - Batch-oriented design
   - Future: Explore streaming architecture

### Areas for Improvement

1. **Documentation**
   - More beginner-friendly tutorials
   - API reference completeness
   - Error message clarity

2. **Error Handling**
   - Better error messages for common issues
   - Validation of user inputs
   - Helpful suggestions for configuration

3. **Testing**
   - Increase test coverage
   - Add integration tests
   - Performance regression tests

4. **Monitoring**
   - Training progress monitoring
   - Inference latency tracking
   - Resource usage metrics

## Collaboration Guidelines

### Contribution Process

1. **Fork and Clone**
   - Fork repository on GitHub
   - Clone forked repository
   - Create feature branch

2. **Development**
   - Activate virtual environment
   - Install with `[dev, notebook]` extras
   - Make changes with type hints and tests

3. **Testing**
   - Run pytest with coverage
   - Ensure all tests pass
   - Check code formatting with black and isort

4. **Submission**
   - Push to feature branch
   - Create pull request
   - Respond to code review comments

### Code Review Priorities

1. **Type Safety**: Proper jaxtyping annotations
2. **Tests**: Adequate test coverage
3. **Documentation**: Clear docstrings and comments
4. **Performance**: No regressions
5. **Style**: Consistent with project standards
