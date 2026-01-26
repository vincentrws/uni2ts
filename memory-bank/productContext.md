# MOIRAI Product Context

## Why This Project Exists

### The Problem

Financial market prediction faces unique challenges that general-purpose time series models fail to address adequately:

1. **OHLCV Semantic Complexity**: Traditional models don't understand that Open, High, Low, Close prices represent the same underlying asset during a time period and should be analyzed collectively.

2. **Normalization Challenges**: OHLC prices should be normalized together, but Volume requires separate normalization due to its different distribution characteristics.

3. **Directional Focus**: Market traders care more about price direction (up/down) than exact price predictions, unlike most forecasting applications.

4. **Market Noise**: High-frequency data contains significant noise, requiring models that can extract meaningful signals from volatile price movements.

5. **Regime Awareness**: Financial markets exhibit different behaviors in bull, bear, and sideways conditions that general models don't recognize.

6. **Corporate Actions**: Stock splits, dividends, and mergers create discontinuous price jumps that confuse standard time series models.

### The Solution

StockMarket-MOIRAI addresses financial forecasting challenges by specializing the universal MOIRAI foundation model:

1. **Fine-tuned Foundation Model**: Leveraging the powerful MOIRAI architecture pre-trained on diverse time series, then adapting it specifically for financial data.

2. **Collective OHLC Normalization**: `CollectiveOHLCScaler` normalizes Open, High, Low, Close prices with shared statistics while handling Volume separately.

3. **Semantic Attention Bias**: `SemanticAttentionBias` enables the model to understand relationships between OHLCV components (Close price driven by price bars, volume provides context).

4. **Directional Optimization**: Focus on directional accuracy metrics and loss functions prioritizing prediction of price movement direction over exact values.

5. **Financial-Specific Architecture**: Adapts mixture distributions and masked encoder training to financial market characteristics and trading horizons.

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

### Quantitative Analysts

**Goal**: Provide sophisticated stock price prediction tools for professional traders

**Experience**:
- Specialized OHLCV forecasting with directional accuracy metrics
- Easy fine-tuning on custom financial datasets
- Backtesting integration for strategy validation
- GPU-optimized inference for real-time trading signals

### Algorithmic Traders

**Goal**: Enable development of ML-powered trading strategies

**Experience**:
- Python API for programmatic strategy development
- OHLCV data pipeline with automatic preprocessing
- Directional focus prioritizing profitable trade signals
- Integration with popular trading platforms and APIs

### Financial Data Scientists

**Goal**: Accelerate financial ML research and development

**Experience**:
- Collective OHLC normalization handling
- Semantic attention bias for multi-variate relationships
- Comprehensive financial evaluation metrics
- Modular architecture for extending with technical indicators

### Fintech Engineers

**Goal**: Build reliable financial prediction systems

**Experience**:
- Production-ready codebase with type safety
- CLI tools optimized for financial workflows
- Model versioning and deployment via HuggingFace Hub
- Clear documentation with trading-specific examples

## Key Features

### 1. Financial Specialization
- Fine-tuned MOIRAI foundation model for OHLCV data
- Collective normalization for OHLC prices
- Semantic understanding of financial relationships

### 2. OHLCV-Centric Architecture
- Handles 5-minute to daily financial data
- Optimized for Close price prediction using full OHLCV context
- Individual Volume processing separate from price data

### 3. Directional Focus
- Directional accuracy metrics for trading applications
- Market regime awareness (bull/bear/sideways conditions)
- Prioritization of price movement over exact values

### 4. Flexible Deployment
- CLI tools optimized for financial workflows
- Python API for trading strategy development
- GluonTS integration with backtesting frameworks

### 5. Financial Evaluation
- Directional accuracy (>50% baseline)
- Market-aware validation (regime-specific performance)
- Backtesting integration for trading strategy validation

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