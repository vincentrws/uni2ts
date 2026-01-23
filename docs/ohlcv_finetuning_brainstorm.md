# Fine-tuning MOIRAI for OHLCV Stock Market Forecasting: Brainstorm Report

## Executive Summary

This report explores the challenges and opportunities of adapting MOIRAI, a universal time series foundation model, for financial market forecasting using 5-minute OHLCV (Open, High, Low, Close, Volume) data. The goal is to create a model that can predict Close price using OHLCV as multivariate context, with focus on directional accuracy rather than exact price paths.

**Key Requirements (Confirmed):**
- **Data**: Parquet files with columns: `ts`, `open`, `high`, `low`, `close`, `volume`
- **Prediction Target**: Close price only, with OHLCV as multivariate context
- **Resolution**: 5-minute bars (primary), other resolutions available
- **Normalization Strategy**: 
  - Volume: Use existing per-column Z-score standardization
  - OHLC: Collective normalization (single mean/std across all 4 price columns)
- **Evaluation**: Focus on directional accuracy (path matters less than direction)
- **Starting Point**: Use existing MOIRAI architecture before making modifications

---

## 1. Domain-Specific Challenges

### 1.1 OHLCV Semantic Structure

OHLCV data has rich semantic relationships:

- **OHLC Interdependence**: All four price columns represent the same underlying asset during a time period
- **Volume-Price Relationship**: Volume provides context for price movements (confirmation vs rejection)
- **Temporal Dependencies**: Each bar builds upon previous information
- **Market Microstructure**: Bid-ask spreads, order book dynamics, liquidity

**Critical Insight**: MOIRAI's existing `PackedStdScaler` already normalizes each variate independently. This is good for volume but we need collective normalization for OHLC.

### 1.2 Normalization Trade-offs

**Current MOIRAI approach**: `PackedStdScaler` computes per-variate mean/std
- **Volume**: This is perfect - use existing scaler
- **OHLC**: Problem - they should be normalized together as they're the same asset's price

**Proposed solution**: Create `CollectiveOHLCScaler` that:
- Computes single mean and std across all 4 OHLC columns
- Normalizes volume independently
- This preserves relative price relationships while removing absolute scale

### 1.3 Market Noise and Signal

**Key observations**:
- **Noise Dominance**: Short-term price movements are largely random
- **Event-driven Dynamics**: News, earnings, economic data create discontinuous jumps
- **Regime Changes**: Market behavior shifts over time
- **Liquidity Effects**: Thinly traded stocks show different patterns

**Strategic implication**: Focus on directional accuracy rather than exact price prediction. The exact path matters less than getting the direction right.

---

## 2. MOIRAI Architecture Analysis

### 2.1 Current Equivariance Assumption

MOIRAI's design has limitations for OHLCV data:

**BinaryAttentionBias** (`src/uni2ts/module/position/attn_bias.py`):
```python
ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
bias = torch.where(ind, self.emb[0], self.emb[1])  # Same vs different
```

**Problem**: Only distinguishes "same variate" vs "different variate" - doesn't understand that Open, High, Low, Close are semantically related price columns.

**Solution**: Add semantic type IDs (close=1, open=2, high=3, low=4, volume=5) and implement `SemanticAttentionBias` that learns relationships between types.

### 2.2 Multi-patch Size Implications

**For 5-minute data**:
- **Small Patches (8-16)**: Capture intraday patterns but may miss longer trends
- **Medium Patches (32-64)**: Balance local detail with broader context
- **Large Patches (128+)**: Smooth out noise but lose granularity

**Recommendation**: Start with patch size 32-64 for 5-minute data. This gives:
- Patch size 32 = 2.67 hours of data per patch
- Context length 512 = ~17 patches ≈ 2 days
- Good balance of temporal resolution and context

### 2.3 Distribution Head Suitability

MOIRAI's mixture distributions:
- **Student-t**: Good for heavy tails in price returns
- **Normal**: May be too restrictive for extreme market moves
- **Log-normal**: Suitable for price data (can't go negative)
- **Mixture Components**: Need to handle both gradual trends and sudden jumps

**Recommendation**: Start with existing mixture distribution - it's designed for diverse data types and should work well for financial data.

### 2.4 Sequence Length Considerations

**Financial context windows**:
- **Short-term (hours)**: Captures intraday patterns, but limited context
- **Medium-term (days)**: Includes market session effects, daily patterns
- **Long-term (weeks)**: Provides trend context but increases noise

**Recommendation**:
- Context length: 512 (≈2 days of 5-minute data)
- Prediction length: 12 (1 hour ahead)
- This gives reasonable context while keeping predictions actionable

---

## 3. Proposed Architecture Modifications

### 3.1 Column Identity Awareness

**Approach**: Extend beyond binary same/different relationships

**Key components**:
1. **AddVariateTypeID transformation**: Assigns semantic type IDs (close=1, open=2, high=3, low=4, volume=5)
2. **SemanticAttentionBias**: Learns type-based attention patterns
3. **Modified attention mechanism**: Accepts and uses type IDs

**Benefits**:
- Model learns that open→close transition is different from high→low
- Volume can have different attention patterns than prices
- Enables future extension with technical indicators

### 3.2 Type-Specific Normalization

**Two-layer approach**:

**Layer 1 - Collective OHLC normalization**:
```python
class CollectiveOHLCScaler(PackedScaler):
    """
    Normalize OHLC columns collectively using single mean and std.
    Volume columns are normalized individually.
    """
    # Compute single mean/std across all OHLC columns
    # Apply same transformation to all OHLC
    # Normalize volume independently
```

**Layer 2 - Volume normalization**:
- Use existing `PackedStdScaler` behavior for volume
- Volume has different distribution than prices
- Standard Z-score is appropriate

### 3.3 Target Specification

**Current behavior**: MOIRAI predicts all variates equally during training

**Our approach**: Keep this unchanged - during training, the model predicts all 5 variates (OHLCV). However, during evaluation:
- Focus metrics on Close price predictions
- Optionally add target-weighted loss (but start simple)
- The OHLCV context still informs Close predictions through masked training

**Rationale**: Predicting all variates during training helps the model learn relationships between them. We just evaluate what we care about (Close).

### 3.4 Market-Aware Embeddings (Optional/Future)

**Potential enhancement**: Add temporal market context

```python
class MarketEmbedding:
    """Add time-of-day and day-of-week embeddings."""
    def forward(self, timestamps):
        # Extract time features
        time_features = {
            'time_of_day': timestamps.hour * 60 + timestamps.minute,
            'day_of_week': timestamps.dayofweek,
            'is_regular_hours': ((9.5 * 60 <= time) & (time < 16 * 60)),
        }
        return embed_features(time_features)
```

**Consideration**: Start without this - add later if needed for performance gains.

---

## 4. Data Pipeline Design

### 4.1 OHLCV Standardization

**Ensure consistent column ordering**: `[open, high, low, close, volume]`

**Important**: This ordering must be consistent across:
- Data loading
- Type ID assignment
- Transformation pipeline
- Inference

**Implementation**: Add validation to ensure correct column order.

### 4.2 Window-Level Normalization Strategy

**For training**:
- Use `CollectiveOHLCScaler` within each training example
- Ensures normalization is relative to context window
- Handles non-stationarity to some degree

**For inference**:
- Same scaler as training (for consistency)
- Normalization parameters learned during training

### 4.3 Missing Data Handling

**Financial data characteristics**:
- Market hours gaps (nights/weekends)
- Corporate actions (splits, dividends)
- Thin trading periods
- Data errors

**Strategy**:
- For 5-minute data, fill missing bars with forward fill
- Mark filled values in observed_mask
- Let the model learn to handle filled data

### 4.4 Sequence Construction

**Training sequences**:
```python
def build_training_sequences(
    data,           # [total_len, 5] OHLCV data
    context_length=512,  # ~2 days
    pred_length=12,      # 1 hour
    stride=1
):
    sequences = []
    for i in range(0, len(data) - context_length - pred_length, stride):
        context = data[i:i+context_length]
        target = data[i+context_length:i+context_length+pred_length]
        sequences.append((context, target))
    return sequences
```

**Key considerations**:
- Stride=1 gives maximum training data
- Use larger stride for faster iteration (stride=pred_length)
- Ensure sequences don't cross market session boundaries

---

## 5. Training Strategy

### 5.1 Fine-tuning Approach

**Leverage pre-trained MOIRAI**:

**Phase 0 - Baseline**:
1. Load pre-trained Moirai-1.1-R-small
2. Fine-tune on OHLCV data without modifications
3. Establish baseline performance
4. Verify data pipeline works

**Phase 1 - Add collective normalization**:
1. Implement `CollectiveOHLCScaler`
2. Add `AddVariateTypeID` transformation
3. Fine-tune again
4. Compare with baseline

**Phase 2 - Add semantic awareness**:
1. Implement `SemanticAttentionBias`
2. Modify attention to use type IDs
3. Fine-tune again
4. Compare with Phase 1

**Phase 3 - Directional optimization** (optional):
1. Add directional accuracy to loss
2. Fine-tune with new loss
3. Compare with Phase 2

### 5.2 Loss Function Extensions

**Start simple**: Use existing NLL loss

**Potential enhancement**: Add directional component

```python
class DirectionalAwareLoss(nn.Module):
    """
    Standard NLL loss + directional accuracy bonus.
    """
    def __init__(self, base_loss, direction_weight=0.1):
        super().__init__()
        self.base_loss = base_loss
        self.direction_weight = direction_weight
    
    def forward(self, pred, target):
        nll_loss = self.base_loss(pred, target)
        
        # Directional accuracy (Close price only)
        pred_close = pred.mean[..., 0]  # Assuming Close is first
        true_close = target[..., 0]
        
        pred_dir = torch.sign(pred_close.diff(dim=-1))
        true_dir = torch.sign(true_close.diff(dim=-1))
        direction_acc = (pred_dir == true_dir).float().mean()
        
        # Combine losses
        return nll_loss - self.direction_weight * direction_acc
```

**Recommendation**: Start without this - add only if directional accuracy is insufficient.

### 5.3 Regularization Strategies

**Prevent overfitting to market noise**:

- **Early stopping**: Monitor validation loss on held-out time period
- **Dropout**: Standard MOIRAI dropout should suffice
- **Weight decay**: Standard MOIRAI weight decay
- **Ensemble**: Average predictions across multiple random seeds

**Critical**: Use time-based validation (train on past, validate on recent) to simulate real deployment.

### 5.4 Curriculum Learning

**Progressive training complexity** (optional, for advanced users):

```python
# Phase 1: Short horizons
context=256, pred=4   # ~1 day, 20 min
learning_rate=1e-4

# Phase 2: Medium horizons  
context=512, pred=12   # ~2 days, 1 hour
learning_rate=5e-5

# Phase 3: Long horizons
context=1024, pred=96  # ~4 days, 8 hours
learning_rate=1e-5
```

**Recommendation**: Start with fixed parameters, add curriculum learning if needed.

---

## 6. Evaluation Framework

### 6.1 Financial Metrics

**Primary metric: Directional Accuracy**

```python
def directional_accuracy(pred, target):
    """
    Percentage of correct up/down predictions.
    Args:
        pred: Predicted Close prices
        target: True Close prices
    Returns:
        Accuracy (0-1)
    """
    pred_direction = torch.sign(pred.diff(dim=-1))
    true_direction = torch.sign(target.diff(dim=-1))
    correct = (pred_direction == true_direction).float()
    return correct.mean().item()
```

**Secondary metrics**:
- **MSE**: Mean squared error (Close price)
- **MAE**: Mean absolute error (Close price)
- **Trend Consistency**: Multi-step directional accuracy

### 6.2 Backtesting Framework (Optional)

**For production validation**: Simulate trading

```python
class SimpleBacktest:
    """
    Very simple trading simulation.
    Buy when predicted up, sell when predicted down.
    """
    def __init__(self, initial_capital=100000, transaction_cost=0.001):
        self.capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position = None
    
    def execute(self, predictions, prices):
        # Convert predictions to signals
        signals = np.sign(predictions.diff())
        
        # Simple strategy
        for i, signal in enumerate(signals):
            if signal > 0 and self.position is None:
                # Buy
                shares = (self.capital * 0.5) / prices[i]
                cost = shares * prices[i] * (1 + self.transaction_cost)
                self.capital -= cost
                self.position = (prices[i], shares)
            elif signal < 0 and self.position is not None:
                # Sell
                entry_price, shares = self.position
                revenue = shares * prices[i] * (1 - self.transaction_cost)
                self.capital += revenue
                self.position = None
        
        return self.capital
```

**Recommendation**: Start with evaluation metrics only, add backtesting if needed for production validation.

### 6.3 Cross-Market Validation

**Test across different conditions**:

- **Bull markets**: 2009-2010, 2020-2021
- **Bear markets**: 2007-2008, 2022
- **Sideways**: 2015-2016
- **High volatility**: 2008-2009, 2020

**Strategy**: Ensure training data includes multiple regimes for robustness.

### 6.4 Robustness Testing

**Stress test the model**:

- **Out-of-sample**: Future periods not seen in training
- **Different assets**: Various stocks, sectors, market caps
- **Market events**: Earnings, Fed announcements, geopolitical events
- **Data quality**: Missing data, erroneous ticks, corporate actions

---

## 7. Implementation Roadmap

### Phase 0: Foundation (Week 1)

**Objectives**:
- Verify data quality and structure
- Test existing MOIRAI without modifications
- Establish baseline performance

**Tasks**:
- [ ] Load and validate parquet files
- [ ] Verify column structure: `ts`, `open`, `high`, `low`, `close`, `volume`
- [ ] Check for missing values and data gaps
- [ ] Create baseline data loader
- [ ] Test fine-tuning on small subset (1-2 stocks)
- [ ] Implement directional accuracy metric
- [ ] Establish baseline performance

**Deliverables**:
- Data validation report
- Working data loader
- Baseline performance metrics

### Phase 1: Collective Normalization (Week 2)

**Objectives**:
- Implement custom OHLC scaler
- Integrate into MOIRAI pipeline
- Verify normalization correctness

**Tasks**:
- [ ] Implement `CollectiveOHLCScaler`
- [ ] Add `AddVariateTypeID` transformation
- [ ] Modify `MoiraiModule` to use new scaler
- [ ] Test normalization on sample data
- [ ] Fine-tune with collective normalization
- [ ] Compare with Phase 0 baseline

**Deliverables**:
- Working collective OHLC scaler
- Performance comparison report

### Phase 2: Semantic Awareness (Weeks 3-4)

**Objectives**:
- Add semantic attention bias
- Enable type-aware learning
- Evaluate improvements

**Tasks**:
- [ ] Implement `SemanticAttentionBias`
- [ ] Modify attention to accept type IDs
- [ ] Update transformer configuration
- [ ] Fine-tune with semantic awareness
- [ ] Compare with Phase 1 baseline
- [ ] Analyze attention patterns

**Deliverables**:
- Semantic attention implementation
- Performance comparison report
- Attention pattern analysis

### Phase 3: Directional Optimization (Week 5 - Optional)

**Objectives**:
- Add directional accuracy to loss
- Optimize for direction rather than exact path

**Tasks**:
- [ ] Implement `DirectionalAwareLoss`
- [ ] Tune direction weight hyperparameter
- [ ] Fine-tune with new loss
- [ ] Compare with Phase 2 baseline

**Deliverables**:
- Directional loss implementation
- Performance comparison

### Phase 4: Production Readiness (Weeks 6-8)

**Objectives**:
- Create training/evaluation scripts
- Add documentation
- Optimize for deployment

**Tasks**:
- [ ] Create training script
- [ ] Create evaluation script
- [ ] Write documentation
- [ ] Create examples
- [ ] Add unit tests
- [ ] Optimize for inference

**Deliverables**:
- Production-ready implementation
- Complete documentation

---

## 8. Risk Assessment

### 8.1 Overfitting Risks

**Risks**:
- Model memorizing specific market patterns
- Poor generalization to new market conditions
- Performance degradation over time

**Mitigation**:
- Time-based cross-validation
- Early stopping
- Regularization
- Ensemble methods
- Continuous monitoring

### 8.2 Market Regime Changes

**Risks**:
- Bull market model failing in bear markets
- Sudden regime shifts causing losses
- Inability to adapt to new dynamics

**Mitigation**:
- Train on diverse conditions
- Regular retraining
- Ensemble of regime-specific models
- Risk management in deployment

### 8.3 Data Quality Issues

**Risks**:
- Missing or erroneous data
- Corporate actions not handled
- Thin trading causing unreliable signals

**Mitigation**:
- Robust validation pipeline
- Data quality monitoring
- Handle corporate actions explicitly
- Filter low-liquidity assets

---

## 9. Success Criteria

### Quantitative Targets

| Metric | Target | Baseline |
|--------|--------|----------|
| Directional Accuracy | >55% | 50% (random) |
| Trend Consistency | >60% | 50% (random) |
| MSE (Close) | <0.01 | TBD |

### Qualitative Achievements

- **Pattern Recognition**: Model learns OHLCV relationships
- **Directional Focus**: Optimized for direction, not exact path
- **Generalization**: Works across different assets
- **Extensibility**: Easy to add technical indicators

---

## 10. Conclusion

Fine-tuning MOIRAI for OHLCV data presents unique opportunities. Key insights:

1. **Start simple**: Use existing MOIRAI architecture first
2. **Progressive improvements**: Add features incrementally
3. **Focus on direction**: Exact path less important than direction
4. **Collective normalization**: OHLC should be normalized together
5. **Semantic awareness**: Model benefits from understanding column types

**Next steps**:
1. Begin Phase: Data validation and baseline
2. Establish baseline performance
3. Add features incrementally
4. Evaluate at each phase

This approach balances innovation with practicality, building on MOIRAI's strong foundation while adding financial domain awareness.

---

**Last Updated**: 2026-01-23
**Status**: Brainstorming Complete - Ready for Implementation