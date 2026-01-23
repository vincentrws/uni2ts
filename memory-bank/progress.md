# MOIRAI Project Progress

## Current Status (January 2026)

**Overall Status**: ✅ Production Ready - Mature and Actively Maintained

The MOIRAI project is in a stable, production-ready state with all core functionality implemented and tested. The codebase is mature, well-documented, and actively maintained by the Salesforce AI Research team.

## What Works

### ✅ Core Functionality

#### Model Implementations
- ✅ **Moirai-1.0-R**: Multi-patch size architecture with mixture distributions
  - Small (14M params), Base (91M params), Large (311M params)
  - Full probabilistic forecasting support
  - Multi-patch size projection layers
- ✅ **Moirai-1.1-R**: Improved version of 1.0-R
  - Enhanced performance and stability
  - All size variants available
- ✅ **Moirai-MoE-1.0-R**: Mixture-of-Experts architecture
  - Small and Base variants
  - Sparse expert routing for efficiency
  - Competitive with larger dense models
- ✅ **Moirai-2.0-R**: Simplified deterministic architecture
  - Small variant released (August 2025)
  - Fixed patch size (16)
  - Deterministic quantile regression output

#### Training Pipelines
- ✅ **Pre-training**: Full pipeline for LOTSA dataset
  - Masked encoder training objective
  - Sequence packing for efficiency
  - Distributed training support via Lightning
  - Hydra configuration management
- ✅ **Fine-tuning**: Comprehensive fine-tuning support
  - Load pre-trained models
  - Custom dataset support
  - Validation split and early stopping
  - Multiple benchmark examples (LSF, Monash, PF)

#### Data Handling
- ✅ **Data Loading**: Multiple dataset formats supported
  - Wide format (pandas DataFrame)
  - Long format (univariate time series)
  - Wide multivariate format
  - HuggingFace dataset integration
- ✅ **Transformations**: Comprehensive preprocessing pipeline
  - Time index addition
  - Observed mask handling
  - Patching (multiple patch sizes)
  - Sequence packing
  - Normalization (std scaling)
  - Imputation support
- ✅ **Data Builder**: Easy dataset preparation
  - Simple command-line tool
  - Automatic train/validation splitting
  - Normalization options
  - Multiple format support

#### Inference & Evaluation
- ✅ **Zero-Shot Forecasting**: Out-of-the-box predictions
  - GluonTS integration
  - Batch inference support
  - Probabilistic sampling (for Moirai-1.x and MoE)
  - Rolling window evaluation
- ✅ **Evaluation Metrics**: Comprehensive benchmarking
  - MSE (Mean Squared Error)
  - MASE (Mean Absolute Scaled Error)
  - CRPS (Continuous Ranked Probability Score)
  - Multiple other metrics supported
- ✅ **Benchmark Integration**: Standard benchmark support
  - Monash Time Series Forecasting
  - Long Sequence Forecasting (LSF)
  - Probabilistic Forecasting (PF)
  - GIFT-Eval leaderboard

### ✅ Developer Experience

#### Command-Line Interface
- ✅ **Training**: `python -m cli.train`
  - Pre-training support
  - Fine-tuning support
  - Hydra configuration override
- ✅ **Evaluation**: `python -m cli.eval`
  - Custom dataset evaluation
  - Benchmark dataset evaluation
  - Flexible metric selection
- ✅ **Data Building**: `python -m uni2ts.data.builder.simple`
  - Easy dataset preprocessing
  - Multiple format support
  - Validation split options

#### Python API
- ✅ **Model Loading**: `from_pretrained()` for all models
- ✅ **Predictor Creation**: `create_predictor()` for inference
- ✅ **GluonTS Integration**: Standard forecasting interface
- ✅ **Type Safety**: jaxtyping annotations throughout

#### Documentation
- ✅ **README.md**: Comprehensive getting started guide
- ✅ **Jupyter Notebooks**: Multiple examples in `example/` directory
  - Basic forecasting
  - Pandas DataFrame usage
  - Data preparation
  - Visualization
- ✅ **Research Papers**: Published papers for all major variants
- ✅ **Code Documentation**: Docstrings in all public APIs

#### Testing
- ✅ **Unit Tests**: Comprehensive test coverage
  - Model tests
  - Data loading tests
  - Transformation tests
  - Distribution tests
  - Loss function tests
- ✅ **Integration Tests**: End-to-end workflow testing
- ✅ **Benchmark Tests**: Performance regression tests
- ✅ **Fixtures**: Reusable test fixtures

### ✅ Infrastructure

#### Version Control
- ✅ Git repository with clear commit history
- ✅ Pre-commit hooks for code quality
- ✅ Semantic versioning

#### Build System
- ✅ Hatch-based build system
- ✅ PyPI package: `pip install uni2ts`
- ✅ Source installation: `pip install -e '.[notebook]'`

#### Model Distribution
- ✅ HuggingFace Hub integration
- ✅ All model variants available
- ✅ Easy model loading: `from_pretrained()`

#### Continuous Integration
- ✅ Pre-commit configuration
- ✅ Code formatting (black, isort)
- ✅ Type checking (mypy, jaxtyping)

## What's Left to Build

### 🚧 High Priority

#### Documentation Enhancements
- ⚠️ More beginner-friendly tutorials
- ⚠️ API reference completeness
- ⚠️ Troubleshooting guide expansion
- ⚠️ Video tutorials or walkthroughs
- ⚠️ OHLCV forecasting tutorial with dynamic features

#### Example Notebooks
- ✅ Fixed moirai_forecast_pandas.ipynb (January 2026)
  - Removed num_samples from Moirai2Forecast (API mismatch)
  - Updated all 4 cells with moirai2 cases
  - Added missing moirai2 case to dynamic features section
- ⚠️ Create OHLCV forecasting example notebook
  - Use past_feat_dynamic_real for historical OHLV
  - Use feat_dynamic_real for calendar/time features
  - Demonstrate no look-ahead bias approach

#### Error Handling
- ⚠️ Better error messages for common issues
- ⚠️ Input validation with helpful suggestions
- ⚠️ Configuration validation
- ⚠️ Graceful degradation for edge cases

### 📋 Medium Priority

#### Performance Optimizations
- ⚠️ ONNX export for optimized CPU inference
- ⚠️ Model quantization for edge deployment
- ⚠️ Flash attention optimization
- ⚠️ Compiled model improvements

#### Feature Additions
- ⚠️ Automated hyperparameter tuning integration
- ⚠️ Multi-modal extensions (text + time series)
- ⚠️ Real-time streaming inference
- ⚠️ Custom distribution plugin system

#### Monitoring & Observability
- ⚠️ Training progress monitoring dashboards
- ⚠️ Inference latency tracking
- ⚠️ Resource usage metrics
- ⚠️ A/B testing framework

### 🔮 Low Priority / Future Work

#### Next-Generation Models
- ⚠️ Latent diffusion architectures
- ⚠️ Causal transformer improvements
- ⚠️ Multi-modal foundation models

#### Enterprise Features
- ⚠️ Scalable serving infrastructure
- ⚠️ Cloud platform guides (AWS, GCP, Azure)
- ⚠️ API service for remote inference
- ⚠️ Marketplace for fine-tuned models

#### Ecosystem Development
- ⚠️ Third-party integrations
- ⚠️ Community plugin system
- ⚠️ Model sharing platform

## Known Issues

### Current Limitations

1. **PyTorch Version Constraints**
   - **Issue**: Limited to PyTorch 2.1.x-2.4.x
   - **Impact**: Cannot use latest PyTorch features
   - **Workaround**: Use supported PyTorch version
   - **Plan**: Evaluate PyTorch 2.5+ compatibility

2. **Context Length Memory**
   - **Issue**: Maximum context limited by GPU memory
   - **Impact**: Very long sequences (>2000 patches) require large GPUs
   - **Workaround**: Use gradient checkpointing or smaller models
   - **Plan**: Investigate memory-efficient attention mechanisms

3. **CPU Inference Performance**
   - **Issue**: Slower than GPU, especially for large batches
   - **Impact**: Limits deployment on CPU-only environments
   - **Workaround**: Use GPU when possible
   - **Plan**: ONNX export for optimized CPU inference

4. **Real-Time Deployment**
   - **Issue**: No streaming inference support
   - **Impact**: Batch-oriented design not suitable for real-time
   - **Workaround**: Use small batches for low-latency requirements
   - **Plan**: Explore streaming architecture

### Minor Issues

1. **Error Message Clarity**
   - Some error messages could be more helpful
   - Lack of suggestions for common configuration mistakes

2. **Documentation Gaps**
   - Some advanced features lack detailed examples
   - API reference not complete for all modules

3. **Test Coverage**
   - Some edge cases not covered by tests
   - Integration tests could be more comprehensive

## Evolution of Project Decisions

### Model Architecture Evolution

#### Moirai-1.0-R (Original)
- **Decision**: Multi-patch size projection layers
- **Rationale**: Handle all frequencies in single model
- **Result**: Successful universal forecasting capability
- **Learnings**: Multi-patch approach is key to universality

#### Moirai-MoE (October 2024)
- **Decision**: Add Mixture-of-Experts variant
- **Rationale**: Improve efficiency with sparse activation
- **Result**: Competitive performance with fewer active parameters
- **Learnings**: MoE is promising for scaling to larger models

#### Moirai-2.0-R (August 2025)
- **Decision**: Simplified architecture with fixed patch size
- **Rationale**: Improve interpretability and inference speed
- **Result**: Faster inference with competitive quality
- **Learnings**: Deterministic quantile regression sufficient for many use cases

### Training Strategy Evolution

#### Initial Approach (March 2024)
- **Decision**: Masked encoder training on LOTSA
- **Rationale**: Learn bidirectional representations for zero-shot
- **Result**: Strong zero-shot performance
- **Learnings**: Large-scale pre-training is essential for universality

#### Sequence Packing
- **Decision**: Pack multiple sequences into single batch
- **Rationale**: Reduce padding waste
- **Result**: Reduced padding from ~61% to <0.4%
- **Learnings**: Efficient data loading critical for large-scale training

#### Fine-tuning Enhancements (August 2025)
- **Decision**: Enhanced fine-tuning module
- **Rationale**: Improve fine-tuning workflow and results
- **Result**: Better fine-tuning performance and user experience
- **Learnings**: Fine-tuning still valuable for domain-specific tasks

### Distribution Strategy Evolution

#### Original Design
- **Decision**: Mixture of four parametric distributions
- **Rationale**: Handle diverse data characteristics
- **Result**: Universal coverage of data types
- **Learnings**: Mixture distributions provide flexibility

#### Moirai-2.0-R Change
- **Decision**: Deterministic quantile regression
- **Rationale**: Faster inference, better interpretability
- **Result**: Competitive performance with simpler architecture
- **Learnings**: Probabilistic output not always necessary

### Evaluation Strategy Evolution

#### Standard Benchmarks (March 2024)
- **Decision**: Support Monash, LSF, PF benchmarks
- **Rationale**: Standardized evaluation across research community
- **Result**: Easy comparison with other models
- **Learnings**: Benchmark integration essential for research

#### GIFT-Eval (November 2024)
- **Decision**: Create general foundation model benchmark
- **Rationale**: Standardize evaluation across different foundation models
- **Result**: Active leaderboard and community engagement
- **Learnings**: Dedicated benchmark helps compare foundation models

### Technology Stack Evolution

#### Initial Stack
- **Decision**: PyTorch + Lightning + GluonTS
- **Rationale**: Leverage existing ML infrastructure
- **Result**: Rapid development and adoption
- **Learnings**: Good foundation for time series forecasting

#### Type Safety Addition
- **Decision**: Add jaxtyping for tensor shape annotations
- **Rationale**: Catch shape errors at development time
- **Result**: Improved code quality and debugging
- **Learnings**: Type safety valuable for complex tensor operations

#### Configuration Management
- **Decision**: Use Hydra for experiment configuration
- **Rationale**: Declarative, reproducible experiments
- **Result**: Easy experiment management and reproducibility
- **Learnings**: Good configuration system essential for research

## Milestones Achieved

### Research Milestones
- ✅ **March 2024**: Initial MOIRAI paper published
- ✅ **May 2024**: MOIRAI paper accepted to ICML 2024 as Oral
- ✅ **October 2024**: Moirai-MoE paper published
- ✅ **November 2024**: GIFT-Eval benchmark released
- ✅ **August 2025**: Moirai-2.0-R released

### Development Milestones
- ✅ **March 2024**: Initial Uni2TS library release
- ✅ **March 2024**: LOTSA dataset public release
- ✅ **June 2024**: Moirai-1.1-R models released
- ✅ **October 2024**: Moirai-MoE models released
- ✅ **August 2025**: Moirai-2.0-R-small released

### Community Milestones
- ✅ **Active GitHub repository** with stars and forks
- ✅ **HuggingFace model downloads** across all variants
- ✅ **Research citations** and academic adoption
- ✅ **Community contributions** via pull requests and issues

## Future Roadmap

### Q1 2026
- [ ] Complete Moirai-2.0-R documentation
- [ ] Benchmark Moirai-2.0-R on standard datasets
- [ ] Improve error messages and validation
- [ ] Expand beginner tutorials

### Q2 2026
- [ ] Investigate PyTorch 2.5+ compatibility
- [ ] ONNX export for optimized CPU inference
- [ ] Model quantization research
- [ ] Community contribution guidelines

### Q3 2026
- [ ] Explore multi-modal extensions
- [ ] Investigate streaming inference
- [ ] Enhanced monitoring and observability
- [ ] Cloud platform deployment guides

### Q4 2026
- [ ] Next-generation model research
- [ ] Enterprise feature development
- [ ] Plugin system for extensions
- [ ] Model marketplace exploration

## Metrics & KPIs

### Code Quality
- ✅ Test coverage: Comprehensive (target: >80%)
- ✅ Type annotation coverage: >90%
- ✅ Documentation coverage: All public APIs documented
- ✅ Code formatting: Consistent (black, isort)

### Performance
- ✅ Zero-shot performance: Competitive with task-specific models
- ✅ Training efficiency: <0.4% padding waste
- ✅ Inference speed: <100ms per sample on GPU (small model)
- ✅ Memory usage: Efficient with sequence packing

### Community
- ✅ GitHub stars: Active engagement
- ✅ HuggingFace downloads: Regular usage
- ✅ Research citations: Academic impact
- ✅ Community contributions: Ongoing

## Success Criteria

### ✅ Met
- [x] Publication in top-tier conference (ICML 2024 Oral)
- [x] Competitive zero-shot performance on benchmarks
- [x] Production-ready codebase with tests
- [x] Active community engagement
- [x] Multiple model variants for different use cases

### 🎯 In Progress
- [ ] Comprehensive beginner documentation
- [ ] Enterprise deployment guides
- [ ] Real-time inference support
- [ ] Multi-modal extensions

### 📋 Future
- [ ] Next-generation model architecture
- [ ] Automated hyperparameter tuning
- [ ] Model marketplace/ecosystem
- [ ] Cloud platform native integration

## Conclusion

The MOIRAI project has successfully achieved its core goals of creating a universal time series forecasting foundation model. The codebase is production-ready, well-maintained, and actively used by the research and practitioner communities. While there are always opportunities for improvement and expansion, the project has reached a mature state where it can reliably support a wide range of time series forecasting tasks.

The focus going forward should be on:
1. Improving user experience (documentation, error handling)
2. Expanding capabilities (new features, optimizations)
3. Growing the community (contributions, ecosystem)
4. Exploring next-generation approaches (multi-modal, streaming)

The strong foundation built by the team provides an excellent platform for continued innovation in time series forecasting.