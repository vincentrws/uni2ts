# MOIRAI Product Context

## Why This Project Exists

### The Problem

Traditional time series forecasting approaches suffer from several fundamental limitations:

1. **Dataset-Specific Models**: Most forecasting models require training on each individual dataset, making them impractical for real-world scenarios where organizations need to forecast thousands of different time series across domains.

2. **Frequency Dependence**: Models trained on one frequency (e.g., hourly data) fail to generalize to other frequencies (e.g., daily, monthly), requiring separate models for each frequency.

3. **Multivariate Complexity**: Handling varying numbers of variables requires different model architectures, preventing a unified approach.

4. **Resource Intensive**: Training specialized models for each use case requires significant computational resources and ML expertise.

5. **Expert Knowledge Barrier**: Effective forecasting typically requires domain experts to select appropriate models, features, and hyperparameters.

### The Solution

MOIRAI addresses these challenges by:

1. **Universal Pre-training**: Training on a massive, diverse dataset (LOTSA with 27B observations) to learn universal time series patterns that generalize across domains.

2. **Frequency Agnostic**: Multi-patch size projection layers enable handling any frequency from yearly to second-level without architectural changes.

3. **Any-Variate Support**: Binary attention bias mechanism allows handling unlimited variables without fixed-dimensional embeddings.

4. **Zero-Shot Deployment**: Pre-trained models can be deployed immediately without task-specific training, dramatically reducing time-to-production.

5. **Probabilistic Flexibility**: Mixture distribution heads handle diverse data characteristics (counts, skewed distributions, stable series) automatically.

### Market Need

Organizations across industries need robust forecasting capabilities:

- **Retail**: Sales forecasting, inventory optimization
- **Energy**: Load forecasting, renewable energy prediction
- **Finance**: Market analysis, risk assessment, OHLCV price forecasting with dynamic features
- **Healthcare**: Patient monitoring, resource planning
- **Manufacturing**: Demand planning, maintenance scheduling
- **Transportation**: Traffic prediction, logistics optimization

Traditional solutions require building and maintaining dozens of specialized models. MOIRAI provides a single foundation model that handles all these scenarios.

## How MOIRAI Works

### Core Architecture

MOIRAI uses a transformer-based architecture with three key innovations:

#### 1. Multi-Patch Size Projection
Different time frequencies require different patch sizes:
- High frequency (seconds/minutes): Large patches (64-128) to reduce attention computation
- Low frequency (yearly/monthly): Small patches (8-32) to preserve temporal detail

The model learns multiple projection layers and dynamically selects appropriate ones based on input frequency.

#### 2. Any-Variate Attention
Variables are flattened into a single sequence with two encoding mechanisms:
- **Temporal encoding**: Rotary Position Embeddings (RoPE) for time index
- **Variate encoding**: Binary attention bias indicating whether patches belong to same/different variables

This design is equivariant to variable ordering and scales to unlimited dimensions.

#### 3. Mixture Distribution Heads
The model outputs a mixture of four distributions:
- **Student's t**: General robust forecasting
- **Negative Binomial**: Count data (sales, traffic)
- **Log-normal**: Right-skewed distributions (economic data)
- **Low-variance Normal**: High-confidence predictions

Each distribution handles different data characteristics, providing comprehensive coverage.

### Training Methodology

#### Masked Encoder Objective
Similar to BERT, MOIRAI uses masked prediction:
1. Randomly mask portions of the input sequence
2. Train the encoder to predict masked values
3. Minimize negative log-likelihood of mixture distribution

#### Dynamic Task Sampling
Unlike fixed-horizon models, MOIRAI samples:
- Variable context length (L)
- Variable prediction horizon (H)
- Total sequence capped at 512 tokens

This enables the model to handle diverse prediction tasks.

#### Sequence Packing
Multiple short time series are packed into single training examples:
- Reduces padding from ~61% to <0.4%
- Dramatically improves training efficiency
- Enables large-batch training on limited GPU memory

### Inference Flow

```
Input Data → GluonTS Dataset → Uni2TS Transformations 
    → Packed Batches → Model Forward Pass 
    → Distribution Sampling → Forecasts
```

The model generates probabilistic forecasts by sampling from the predicted mixture distribution.

## User Experience Goals

### Researchers

**Goal**: Enable rapid experimentation with universal time series forecasting

**Experience**:
- Easy installation: `pip install uni2ts`
- Pre-trained models ready for zero-shot evaluation
- Comprehensive benchmarking tools
- Extensible architecture for novel research

### Data Scientists

**Goal**: Provide production-ready forecasting without ML expertise

**Experience**:
- Simple API: Load model, pass data, get forecasts
- Automatic handling of frequencies and multivariate data
- Built-in evaluation metrics
- Clear visualization tools

### Engineers

**Goal**: Integrate forecasting into production systems

**Experience**:
- Stable, well-tested codebase
- Batch inference capabilities
- Model versioning via HuggingFace Hub
- Clear documentation for deployment

### Students

**Goal**: Learn advanced time series forecasting techniques

**Experience**:
- Clear code structure with extensive type hints
- Jupyter notebook examples
- Step-by-step tutorials
- Links to research papers

## Key Features

### 1. Zero-Shot Forecasting
- No training required for new datasets
- Competitive with task-specific models
- Immediate deployment possible

### 2. Universal Coverage
- Any frequency: Yearly to second-level
- Any variates: Univariate to highly multivariate
- Any domain: Energy, transport, climate, finance, healthcare, etc.

### 3. Probabilistic Predictions
- Full distributional forecasts
- Quantile predictions for risk assessment
- Confidence intervals automatically provided

### 4. Flexible Deployment
- CLI tools for batch processing
- Python API for programmatic usage
- GluonTS integration for compatibility

### 5. Comprehensive Evaluation
- Standard metrics: MSE, MASE, CRPS
- Benchmark suites: Monash, LSF, PF
- Rolling window evaluation
- Comparison with baselines

## Success Metrics

### Research Success
- Publication acceptance at top conferences
- Benchmark performance competitiveness
- Citation count and academic adoption

### User Success
- Active GitHub stars and forks
- HuggingFace model downloads
- Community contributions
- Usage in production systems

### Technical Success
- Zero-shot performance vs. fine-tuned models
- Training efficiency (throughput, memory usage)
- Inference speed for production use
- Code quality (test coverage, documentation)

## Competitive Advantages

1. **True Universality**: Unlike competitors requiring frequency-specific models, MOIRAI handles all frequencies in one model

2. **Any-Variate Scaling**: No fixed limit on number of variables, unlike models with learned embeddings

3. **Production Ready**: Comprehensive testing, type safety, and documentation not found in research-only codebases

4. **Active Development**: Regular updates with new models (Moirai-MoE, Moirai-2.0) and features

5. **Open Source**: Apache 2.0 license enabling commercial use and modification

## Limitations and Considerations

1. **Compute Requirements**: Large models require GPUs for training and efficient inference

2. **Research Only**: Models not evaluated for all downstream purposes, require user validation

3. **Data Preparation**: Custom data must be preprocessed to compatible format

4. **Context Limits**: Maximum context length determined by model architecture

5. **Domain Specificity**: While universal, may still benefit from domain-specific fine-tuning in some cases