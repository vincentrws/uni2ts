# MOIRAI Project Brief

## Project Identity

**Name**: StockMarket-MOIRAI (Fine-tuned MOIRAI for OHLCV Price Forecasting)

**Organization**: Independent Project (Forked from Salesforce AI Research)

**Type**: Specialized Financial Time Series Forecasting Model

**Primary Language**: Python 3.10+

**Framework**: PyTorch + MOIRAI Foundation Model

## Core Purpose

This project fine-tunes the MOIRAI universal time series foundation model specifically for predicting stock market prices using OHLCV (Open, High, Low, Close, Volume) data. It adapts the general-purpose forecasting architecture to understand financial market semantic relationships, with particular focus on directional price movement accuracy for trading applications.

## Vision

To create a specialized financial forecasting model that excels at stock market prediction by leveraging the universal foundation of MOIRAI while incorporating financial-domain specific knowledge about OHLCV relationships, normalization strategies, and market-aware attention mechanisms.

## Core Requirements

### Functional Requirements

1. **OHLCV Forecasting Capability**
   - Specialized prediction of Close price using OHLCV context
   - Handle 5-minute to daily frequency financial data
   - Collective normalization for OHLC prices (shared mean/std across Open, High, Low, Close)
   - Individual normalization for Volume

2. **Financial Market Adaptations**
   - Semantic attention bias understanding OHLCV relationships
   - Directional accuracy optimization over exact price prediction
   - Support for financial data gaps and corporate actions
   - Market regime awareness (bull/bear/sideways markets)

3. **Fine-tuning Infrastructure**
   - Load pre-trained MOIRAI models from HuggingFace Hub
   - Custom fine-tuning on OHLCV datasets
   - Incorporation of CollectiveOHLCScaler and SemanticAttentionBias
   - Support for masked encoder training objective

4. **Financial Evaluation & Inference**
   - Directional accuracy metrics (>50% baseline)
   - Rolling window evaluation on financial time series
   - Integration with GluonTS for production inference
   - Backtesting framework for trading strategy validation

5. **Financial Data Handling**
   - Parquet file format support for OHLCV data
   - Validation of column structure (ts, open, high, low, close, volume)
   - Handling of market hours gaps and missing data
   - Corporate action adjustments and data quality checks

### Non-Functional Requirements

1. **Performance**
   - Efficient fine-tuning with optimized batch sizes for financial data
   - GPU acceleration for real-time inference
   - Memory optimization for high-frequency data processing

2. **Flexibility**
   - Modular design allowing component swapping for financial features
   - Configurable via Hydra YAML files for different market types
   - Extensible architecture for additional technical indicators

3. **Usability**
   - Command-line interface optimized for financial workflows
   - Python API for programmatic trading strategy development
   - Jupyter notebook examples for financial analysis
   - Clear documentation with financial market examples

4. **Reproducibility**
   - Fixed seeds for deterministic financial model training
   - Version control integration for model deployment
   - Model checkpointing for production deployment

## Success Criteria

1. **Financial Prediction Accuracy**
   - Directional accuracy >55% on OHLCV data (vs 50% random baseline)
   - Consistent performance across different market regimes (bull/bear/sideways)
   - Superior directional accuracy compared to unmodified MOIRAI

2. **Practical Utility**
   - Effective OHLCV normalization strategy (CollectiveOHLCScaler)
   - Semantic attention bias implementation for financial understanding
   - Successful deployment in financial trading strategies
   - Active community contribution and financial domain adoption

3. **Code Quality**
   - Comprehensive test coverage for financial components
   - Type safety with jaxtyping for tensor operations
   - Clean modular architecture with financial-specific modules
   - Comprehensive documentation with financial examples

## Target Users

1. **Quantitative Analysts**: Financial professionals needing advanced stock price prediction
2. **Algorithmic Traders**: Developers creating automated trading strategies
3. **Financial Data Scientists**: Practitioners analyzing market data with ML
4. **Finance Researchers**: Academics studying market prediction and forecasting
5. **Fintech Engineers**: Developers building financial technology solutions

## Key Constraints

1. **Technical**
   - Python 3.10+ required
   - PyTorch 2.1-2.4 compatibility
   - GPU support for fine-tuning and inference
   - Financial data processing memory requirements

2. **Data**
   - OHLCV data in Parquet format with consistent column structure
   - Handling of financial market gaps (nights/weekends)
   - Data quality checks for erroneous bars and corporate actions
   - Market condition diversity (bull/bear/sideways periods)

3. **Regulatory & Ethical**
   - Financial regulatory compliance for trading applications
   - Ethical use in algorithmic trading systems
   - Transparency requirements for financial models
   - Risk management considerations

## Project Scope

### In Scope
- Fine-tuning MOIRAI on OHLCV stock data
- Collective OHLC normalization implementation
- Semantic attention bias for financial relationships
- Directional accuracy metrics and evaluation
- Backtesting framework for trading strategies
- Financial data processing pipeline
- Integration with trading platforms and APIs

### Out of Scope
- Real-time market trading execution
- Multi-modal financial analysis (news/text + OHLCV)
- Automated deployment to trading infrastructures
- Regulatory compliance frameworks
- Edge device deployment for retail traders

## Related Work

- **Chronos**: T5-based time series forecasting model
- **TimesFM**: Google's time series foundation model
- **VisionTS**: Vision transformer for time series
- **GIFT-Eval**: General time series forecasting benchmark
- **Financial Domain**: QLSTM, LSTM-based trading models, ARIMA financial variants

## Current Status

The project is in active development focusing on financial specialization:
- Research papers published on original MOIRAI foundation model
- Forked repository with OHLCV fine-tuning capabilities
- Implementation of CollectiveOHLCScaler and SemanticAttentionBias planned
- Active development of financial evaluation metrics
- Integration with trading strategy backtesting frameworks
