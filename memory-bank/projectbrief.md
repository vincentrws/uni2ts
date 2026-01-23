# MOIRAI Project Brief

## Project Identity

**Name**: MOIRAI (Masked Encoder-based UnIveRsAl TIme Series Forecasting Transformer)

**Organization**: Salesforce AI Research

**Type**: Universal Time Series Foundation Model Research Library

**Primary Language**: Python 3.10+

**Framework**: PyTorch

## Core Purpose

MOIRAI is a production-grade PyTorch library for universal time series forecasting that implements a transformer-based foundation model. It enables zero-shot forecasting across diverse domains, frequencies, and multivariate scenarios without requiring task-specific training.

## Vision

To establish a universal time series forecasting model that can handle any frequency, any number of variates, and diverse distributional properties in a zero-shot capacity, democratizing access to state-of-the-art time series forecasting capabilities.

## Core Requirements

### Functional Requirements

1. **Universal Forecasting Capability**
   - Support any frequency (yearly to second-level)
   - Handle any number of variates (univariate to highly multivariate)
   - Support diverse distributional properties through mixture distributions

2. **Model Variants**
   - Moirai-1.0-R: Original multi-patch architecture
   - Moirai-MoE-1.0-R: Mixture-of-Experts variant
   - Moirai-2.0-R: Simplified deterministic architecture

3. **Training Capabilities**
   - Pre-training on LOTSA dataset (27B+ observations)
   - Fine-tuning on custom datasets
   - Support for masked encoder training objective

4. **Inference & Evaluation**
   - Zero-shot forecasting without fine-tuning
   - Rolling window evaluation
   - Comprehensive metrics (MSE, MASE, CRPS, etc.)
   - Integration with GluonTS for inference

5. **Data Handling**
   - Support for wide and long format data
   - Automatic data preprocessing and transformation
   - Efficient sequence packing for training
   - Integration with HuggingFace datasets

### Non-Functional Requirements

1. **Performance**
   - Efficient training with <0.4% padding waste via sequence packing
   - Scalable to billions of observations
   - Support for distributed training

2. **Flexibility**
   - Modular design allowing component swapping
   - Configurable via Hydra YAML files
   - Extensible architecture for new distributions/transforms

3. **Usability**
   - Command-line interface for common tasks
   - Python API for programmatic usage
   - Jupyter notebook examples
   - Clear documentation and type hints

4. **Reproducibility**
   - Fixed seeds for deterministic training
   - Version control integration
   - HuggingFace Hub for model weights

## Success Criteria

1. **Research Impact**
   - Publication in top-tier conferences (ICML 2024 Oral)
   - Competitive performance on standard benchmarks (Monash, LSF, PF)
   - Citation and adoption by research community

2. **Practical Utility**
   - Zero-shot performance competitive with task-specific models
   - Successful deployment in production scenarios
   - Active community contribution and usage

3. **Code Quality**
   - Comprehensive test coverage
   - Type safety with jaxtyping
   - Clean modular architecture
   - Extensive documentation

## Target Users

1. **Researchers**: Time series forecasting researchers exploring foundation models
2. **Data Scientists**: Practitioners needing robust forecasting solutions
3. **Engineers**: Developers integrating forecasting into production systems
4. **Students**: Learners studying advanced time series forecasting techniques

## Key Constraints

1. **Technical**
   - Python 3.10+ required
   - PyTorch 2.1-2.4 compatibility
   - GPU/TPU support for training large models
   - Substantial compute resources for pre-training

2. **Data**
   - LOTSA dataset requires significant storage (~27B observations)
   - Custom data must be preprocessed to compatible format
   - Path management via `.env` file

3. **Licensing**
   - Apache 2.0 license
   - Research use only (not evaluated for all downstream purposes)
   - Ethical considerations for deployment

## Project Scope

### In Scope
- Core transformer model implementations
- Training and evaluation pipelines
- Data loading and preprocessing
- CLI tools for common workflows
- Example notebooks and documentation
- Model weights distribution via HuggingFace

### Out of Scope
- Real-time streaming inference
- Multi-modal extensions (text/image integration)
- Automated hyperparameter tuning
- Production deployment tooling
- Edge device optimization

## Related Work

- **Chronos**: T5-based time series forecasting model
- **TimesFM**: Google's time series foundation model
- **VisionTS**: Vision transformer for time series
- **GIFT-Eval**: General time series forecasting benchmark

## Current Status

The project is mature and actively maintained with:
- Multiple published papers (Moirai, Moirai-MoE, GIFT-Eval)
- Production-ready codebase
- Active community engagement
- Regular model updates (Moirai-1.1-R, Moirai-2.0-R)