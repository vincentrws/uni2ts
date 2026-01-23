# OHLCV Implementation Plan for MOIRAI

## Overview

This document provides a practical, phased implementation plan for adapting MOIRAI to predict Close prices using OHLCV (Open, High, Low, Close, Volume) data as **covariate features**.

## ⚠️ CRITICAL ARCHITECTURAL DECISION (January 2026)

### Use Dynamic Features, NOT Multivariate Forecasting

**Correct Approach**: Univariate target with dynamic covariates
- **Target**: Close price only (`target_dim=1`)
- **Past Features**: Historical OHLV (lookback, no look-ahead bias)
- **Future Features**: Calendar/time features known in advance

**Why NOT Multivariate?**
- Multivariate forecasting predicts ALL variables (O, H, L, C, V) simultaneously
- Would require future OHLV values (look-ahead bias)
- Market data doesn't provide future OHLV - only past values are known at inference time
- Violates causal forecasting constraints

**Why Dynamic Features?**
- Uses only information available at time T (no leakage)
- Model learns patterns: past OHLV → future Close
- Future-known calendar features add seasonal context
- Zero look-ahead bias, production-ready

---

## Data Architecture

### Data Format

**Long Format (Required for covariates)**:
```python
{
    'target': closes,                    # Close prices (target)
    'past_feat_dynamic_real_0': opens,   # Past Open (historical only)
    'past_feat_dynamic_real_1': highs,   # Past High  
    'past_feat_dynamic_real_2': lows,    # Past Low
    'past_feat_dynamic_real_3': volumes, # Past Volume
    'feat_dynamic_real_0': hour_of_day,  # Hour (future-known)
    'feat_dynamic_real_1': day_of_week,  # Day of week (future-known)
    'feat_dynamic_real_2': is_holiday,   # Holiday flag (future-known)
    'item_id': 'stock_symbol'
}
```

### Feature Types Explained

#### `past_feat_dynamic_real` (Historical Only)
**Time-varying features available ONLY in the past** (up to time T)

**Data shape**: `(batch, past_length, num_features)`
- **Only contains historical data** - nothing for future prediction window
- **Examples**:
  - Historical OHLV bars (lagged)
  - Technical indicators computed from historical data (RSI, MACD, etc.)
  - Moving averages, volatility measures
  - Volume-based features

**Example OHLCV past features**:
```python
past_feat_dynamic_real_0: past_open (lookback 100 bars)
past_feat_dynamic_real_1: past_high (lookback 100 bars)
past_feat_dynamic_real_2: past_low (lookback 100 bars)
past_feat_dynamic_real_3: past_volume (lookback 100 bars)
```

#### `feat_dynamic_real` (Future-Known)
**Time-varying features known for the ENTIRE prediction window** (past + future)

**Data shape**: `(batch, past_length + prediction_length, num_features)`
- **Past portion**: historical values
- **Future portion**: **pre-computed future values** (no leakage!)
- **Examples**:
  - Calendar features: hour_of_day, day_of_week, month
  - Temporal features: minutes_since_market_open, trading_session
  - Known schedules: holiday flags, earnings announcements
  - Market structure features: market_hours, trading_calendar

**Example future-known features**:
```python
feat_dynamic_real_0: hour_of_day (0-23)
feat_dynamic_real_1: day_of_week (0-6)
feat_dynamic_real_2: is_market_open (0/1)
feat_dynamic_real_3: is_holiday (0/1)
```

---

## Phase 0: Data Validation & Baseline (Week 1)

### Objectives
- Verify data quality and structure
- Establish correct covariate-based data pipeline
- Test baseline MOIRAI-2.0-R with dynamic features

### Tasks

#### 0.1 Data Validation
- [ ] Load sample parquet files from `data/processed_equities/5m/`
- [ ] Verify columns: `ts`, `open`, `high`, `low`, `close`, `volume`
- [ ] Check for missing values and data gaps
- [ ] Validate timestamp continuity and frequency (5-minute)
- [ ] Confirm data range and quality

**Expected Output**: Data validation report.

#### 0.2 Data Pipeline Prototype
**Critical**: Use **LONG format** for covariates, NOT wide format

```python
import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset

# Load parquet data
df = pd.read_parquet('data/processed_equities/5m/AAPL_5m.parquet')

# Prepare long format DataFrame
df_long = pd.DataFrame({
    'target': df['close'].values,
    'past_feat_dynamic_real_0': df['open'].shift(1).fillna(0).values,  # Lagged Open
    'past_feat_dynamic_real_1': df['high'].shift(1).fillna(0).values,  # Lagged High
    'past_feat_dynamic_real_2': df['low'].shift(1).fillna(0).values,   # Lagged Low
    'past_feat_dynamic_real_3': df['volume'].shift(1).fillna(0).values, # Lagged Volume
    'feat_dynamic_real_0': df['ts'].dt.hour.values,  # Hour (future-known)
    'feat_dynamic_real_1': df['ts'].dt.dayofweek.values,  # Day of week
    'feat_dynamic_real_2': (df['ts'].dt.hour >= 9) & (df['ts'].dt.hour < 16),  # Market hours
    'item_id': 'AAPL'
})

# Create PandasDataset
ds = PandasDataset.from_long_dataframe(
    df_long,
    target='target',
    past_feat_dynamic_real=[
        'past_feat_dynamic_real_0',
        'past_feat_dynamic_real_1', 
        'past_feat_dynamic_real_2',
        'past_feat_dynamic_real_3'
    ],
    feat_dynamic_real=[
        'feat_dynamic_real_0',
        'feat_dynamic_real_1',
        'feat_dynamic_real_2'
    ],
    item_id='item_id'
)
```

- [ ] Implement long format conversion script
- [ ] Test with sample data
- [ ] Verify feature shapes and dimensions

**Expected Output**: Working long format data loader.

#### 0.3 Baseline Test with Moirai2

```python
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from gluonts.dataset.split import split

# Split dataset
train, test_template = split(ds, offset=-100)  # Last 100 bars for test
test_data = test_template.generate_instances(
    prediction_length=12,  # Predict 12 steps ahead (1 hour for 5-min bars)
    windows=8,
    distance=12,
)

# Initialize Moirai-2.0-R (no num_samples parameter!)
model = Moirai2Forecast(
    module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
    prediction_length=12,
    context_length=200,  # 200 bars lookback
    target_dim=1,  # Close only
    feat_dynamic_real_dim=3,  # 3 future-known features
    past_feat_dynamic_real_dim=4,  # 4 historical OHLV features
)

# Create predictor and test
predictor = model.create_predictor(batch_size=32)
forecasts = predictor.predict(test_data.input)
```

- [ ] Run baseline test
- [ ] Verify no errors
- [ ] Check forecast shape and quality

**Expected Output**: Successfully running baseline with dynamic features.

#### 0.4 Baseline Evaluation
- [ ] Implement metrics: MSE, MAE, Directional Accuracy
- [ ] Evaluate on test set
- [ ] Document baseline performance

**Expected Output**: Baseline performance metrics.

---

## Phase 1: Feature Engineering (Week 2)

### Objectives
- Add rich historical features from OHLCV
- Add calendar and market structure features
- Test feature contributions

### Tasks

#### 1.1 Enhanced Historical Features

**Technical Indicators**:
```python
def add_technical_features(df, lookback=100):
    """Compute technical indicators from OHLCV."""
    features = {}
    
    # Returns
    features['return_1'] = df['close'].pct_change(1)
    features['return_5'] = df['close'].pct_change(5)
    features['return_20'] = df['close'].pct_change(20)
    
    # Moving averages
    features['ma_10'] = df['close'].rolling(10).mean()
    features['ma_50'] = df['close'].rolling(50).mean()
    features['ma_ratio'] = features['ma_10'] / features['ma_50']
    
    # Volatility
    features['volatility_20'] = df['close'].rolling(20).std()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume-based features
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['volume_volatility'] = df['volume'].rolling(20).std()
    
    return features
```

- [ ] Implement technical indicator functions
- [ ] Add 10-20 historical features
- [ ] Test feature computation

**Expected Output**: Rich set of historical OHLV features.

#### 1.2 Enhanced Calendar Features

```python
def add_calendar_features(df):
    """Add calendar and market structure features."""
    features = {}
    
    # Temporal features
    features['hour'] = df['ts'].dt.hour
    features['day_of_week'] = df['ts'].dt.dayofweek
    features['day_of_month'] = df['ts'].dt.day
    features['month'] = df['ts'].dt.month
    features['quarter'] = df['ts'].dt.quarter
    
    # Market structure
    features['is_market_open'] = ((df['ts'].dt.hour >= 9) & 
                                   (df['ts'].dt.hour < 16)).astype(int)
    features['is_trading_day'] = df['ts'].dt.dayofweek < 5  # Mon-Fri
    features['minutes_since_open'] = (df['ts'].dt.hour * 60 + 
                                       df['ts'].dt.minute)
    
    # Categorical encodings (one-hot style)
    for hour in range(24):
        features[f'hour_{hour}'] = (df['ts'].dt.hour == hour).astype(int)
    
    for dow in range(5):  # Trading days only
        features[f'dow_{dow}'] = (df['ts'].dt.dayofweek == dow).astype(int)
    
    return features
```

- [ ] Implement calendar feature functions
- [ ] Add holiday database
- [ ] Test feature coverage

**Expected Output**: Comprehensive calendar features.

#### 1.3 Feature Selection & Validation
- [ ] Compute feature importance (ablation test)
- [ ] Remove redundant features
- [ ] Validate no data leakage
- [ ] Test with model

**Expected Output**: Optimized feature set.

---

## Phase 2: Fine-tuning & Optimization (Weeks 3-4)

### Objectives
- Fine-tune Moirai-2.0-R on OHLCV data
- Optimize hyperparameters
- Evaluate directional accuracy

### Tasks

#### 2.1 Fine-tuning Setup

```python
# Fine-tuning configuration
# File: cli/conf/finetune/data/ohlcv.yaml

data:
  _target_: uni2ts.data.builder.simple.SimpleFinetuneDatasetBuilder
  dataset_path: "data/processed_equities/5m/"
  mode: "S"  # Univariate target
  context_length: 500  # ~2 days for 5-min bars
  prediction_length: 12  # 1 hour ahead
  normalize: True
  
model:
  _target_: uni2ts.model.moirai.MoiraiForecast
  module:
    _target_: lightning.pytorch.core.module.Module.load_from_checkpoint
    checkpoint_path: "path/to/moirai-1.1-R-small.ckpt"
  
trainer:
  max_epochs: 50
  learning_rate: 1e-4
  batch_size: 32
  early_stopping_patience: 10
```

- [ ] Create Hydra config
- [ ] Implement fine-tuning script
- [ ] Test on small dataset

#### 2.2 Hyperparameter Tuning

**Key hyperparameters to tune**:
- Context length: 200, 500, 1000
- Prediction length: 12, 24, 48 (1h, 2h, 4h for 5-min bars)
- Learning rate: 1e-5, 5e-5, 1e-4, 5e-4
- Batch size: 16, 32, 64
- Regularization: dropout, weight decay

- [ ] Grid search or random search
- [ ] Use validation set for selection
- [ ] Track metrics

**Expected Output**: Optimized hyperparameters.

#### 2.3 Advanced Evaluation Metrics

```python
def directional_accuracy(pred, target):
    """Percentage of correct up/down predictions."""
    pred_dir = np.sign(pred[1:] - pred[:-1])
    true_dir = np.sign(target[1:] - target[:-1])
    return (pred_dir == true_dir).mean()

def trend_consistency(pred, target, window=5):
    """Trend consistency over window."""
    pred_trend = np.sign(pred[window:] - pred[:-window])
    true_trend = np.sign(target[window:] - target[:-window])
    return (pred_trend == true_trend).mean()

def sharpe_ratio(pred, target, risk_free_rate=0.02):
    """Simulated Sharpe ratio based on predictions."""
    # Simple strategy: buy if predicted up, sell if predicted down
    positions = np.sign(pred[1:] - pred[:-1])
    returns = positions * (target[1:] - target[:-1]) / target[:-1]
    excess_returns = returns - risk_free_rate / 252  # Annualized
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
```

- [ ] Implement financial metrics
- [ ] Add to evaluation pipeline
- [ ] Track directional accuracy

**Expected Output**: Comprehensive evaluation metrics.

#### 2.4 Compare Model Variants

- [ ] Moirai-1.1-R-small (probabilistic)
- [ ] Moirai-2.0-R-small (deterministic, faster)
- [ ] Test both models
- [ ] Document trade-offs

**Expected Output**: Model variant comparison.

---

## Phase 3: Production Implementation (Weeks 5-6)

### Objectives
- Create production-ready inference pipeline
- Optimize for speed and memory
- Add monitoring and logging

### Tasks

#### 3.1 Inference Script

```python
# scripts/infer_ohlcv.py

import pandas as pd
import torch
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from gluonts.dataset.pandas import PandasDataset

def load_and_predict(model_path, data_path, horizon=12):
    """Load model and predict Close prices."""
    
    # Load model
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(model_path),
        prediction_length=horizon,
        context_length=500,
        target_dim=1,
        feat_dynamic_real_dim=10,
        past_feat_dynamic_real_dim=20,
    )
    
    # Load data
    df = pd.read_parquet(data_path)
    df_long = prepare_features(df)  # Your feature preparation
    
    # Create dataset
    ds = PandasDataset.from_long_dataframe(df_long, ...)
    
    # Predict
    predictor = model.create_predictor(batch_size=32)
    forecasts = predictor.predict(ds)
    
    return forecasts

if __name__ == "__main__":
    forecasts = load_and_predict(
        "Salesforce/moirai-2.0-R-small",
        "data/processed_equities/5m/AAPL_5m.parquet"
    )
    
    # Get latest prediction
    pred = list(forecasts)[-1]
    print(f"Prediction: {pred.mean}")