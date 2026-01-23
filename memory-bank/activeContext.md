# MOIRAI Active Context

## Current Work Focus

As of January 2026, the MOIRAI project is in a mature, production-ready state with active maintenance and ongoing research.

### Active Projects

#### 1. Moirai-2.0-R Development
- **Status**: Released (August 2025)
- **Model Size**: Small variant released
- **Architecture**: Simplified from Moirai-1.0-R with fixed patch size (16)
- **Output**: Deterministic quantile regression (no num_samples parameter)
- **Focus**: Improved interpretability and performance

#### 2. OHLCV Forecasting Implementation
- **Status**: In Planning (January 2026)
- **Approach**: Univariate target (Close) with dynamic covariates (OHLV)
- **Architecture**: feat_dynamic_real and past_feat_dynamic_real features
- **Key Decision**: NOT multivariate forecasting - uses covariate features instead
- **Rationale**: Avoids look-ahead bias, uses past OHLV to predict future Close

#### 2. Moirai-MoE Research
- **Status**: Published (October 2024)
- **Model Variants**: Small, Base
- **Architecture**: Mixture-of-Experts for sparse activation
- **Key Innovation**: Expert routing for improved efficiency
- **Performance**: Competitive with larger dense models

#### 3. GIFT-Eval Benchmark
- **Status**: Active (November 2024)
- **Purpose**: General time series forecasting model evaluation
- **Platform**: HuggingFace Spaces leaderboard
- **Goal**: Standardized evaluation across foundation models

## Recent Changes

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

### Immediate Priorities

1. **Documentation Enhancement**
   - Expand tutorial documentation for new users
   - Add more real-world use case examples
   - Improve API reference documentation

2. **Model Performance**
   - Benchmark Moirai-2.0-R on standard datasets
   - Compare performance across all model variants
   - Publish comprehensive results

3. **Community Engagement**
   - Respond to GitHub issues and PRs
   - Encourage community contributions
   - Update GIFT-Eval leaderboard with new models

### Medium-Term Goals

1. **Feature Expansion**
   - Explore multi-modal extensions (text + time series)
   - Investigate real-time streaming inference
   - Add automated hyperparameter tuning

2. **Model Optimization**
   - Implement model quantization for edge deployment
   - Optimize for CPU inference
   - Reduce model size while maintaining performance

3. **Integration**
   - Official support for cloud platforms (AWS, GCP, Azure)
   - Containerized deployment guides
   - API service for remote inference

### Long-Term Vision

1. **Next-Generation Models**
   - Explore latent diffusion architectures
   - Investigate causal transformer improvements
   - Multi-modal time series foundation models

2. **Enterprise Features**
   - Scalable serving infrastructure
   - Monitoring and observability tools
   - A/B testing framework for model comparison

3. **Ecosystem Development**
   - Third-party integrations
   - Plugin system for custom distributions
   - Marketplace for fine-tuned models

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

1. **Forgetting Virtual Environment**
   - Error: ModuleNotFoundError, import errors
   - Solution: Always activate venv before commands
   - Tip: Add reminder to shell prompt or git hook

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

### Issue Resolution

1. **Bug Reports**: Provide minimal reproducible example
2. **Feature Requests**: Explain use case and benefits
3. **Questions**: Search existing issues first
4. **Documentation**: Specify what's unclear or missing

## External Resources

### Official Documentation
- GitHub: https://github.com/SalesforceAIResearch/uni2ts
- HuggingFace Models: https://huggingface.co/collections/Salesforce/moirai-r-models-65c8d3a94c51428c300e0742
- GIFT-Eval Leaderboard: https://huggingface.co/spaces/Salesforce/GIFT-Eval

### Research Papers
- Moirai (ICML 2024): https://arxiv.org/abs/2402.02592
- Moirai-MoE: https://arxiv.org/abs/2410.10469
- GIFT-Eval: https://arxiv.org/abs/2410.10393

### Related Projects
- GluonTS: https://github.com/awslabs/gluonts
- PyTorch Lightning: https://pytorchlightning.ai/
- Hydra: https://hydra.cc/

## Support Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community support
- **Email**: Contact Salesforce AI Research team (for enterprise inquiries)

## Ethical Considerations

MOIRAI is released for research purposes only. Users should:
- Evaluate models for specific use cases
- Consider accuracy, safety, and fairness implications
- Comply with applicable laws and regulations
- Implement safeguards for high-risk applications
- Follow AI usage policies and best practices

See AI_ETHICS.md for detailed guidance.