# OHLCV Adaptation Strategy for MOIRAI

## Executive Summary

This document outlines a comprehensive strategy for adapting the MOIRAI (Masked Encoder-based UnIveRsAl TIme Series Forecasting Transformer) architecture to handle OHLCV (Open, High, Low, Close, Volume) financial data while maintaining extensibility for future technical indicators. The core innovation is a **Semantic Variate ID System** that assigns unique identifiers to each data type, enabling the model to learn semantic relationships between variates while allowing non-destructive expansion for new indicators.

**Key Clarifications (Based on User Requirements):**
- **Data Format**: Parquet files with columns: `ts`, `open`, `high`, `low`, `close`, `volume`
- **Prediction Target**: Close price only, with OHLCV as multivariate context
- **Resolution**: 5-minute bars (primary), other resolutions available
- **Normalization Strategy**: 
  - Volume: Use existing per-column Z-score standardization
  - OHLC: Collective normalization (single mean/std across all 4 price columns)
- **Evaluation**: Focus on directional accuracy (path matters less than direction)
- **Starting Point**: Use existing MOIRAI architecture before making modifications

**Key Objectives:**
- Transform MOIRAI from a generalist to a specialized financial forecasting model
- Enable incremental addition of technical indicators without retraining the core architecture
- Maintain MOIRAI's universal forecasting capabilities while adding domain-specific awareness
- Support efficient fine-tuning with minimal computational overhead

---

## 1. Background: MOIRAI Architecture

### 1.1 Core Concepts

MOIRAI is a Universal Time Series Model (UTSM) pre-trained on the LOTSA archive (27B observations) using a masked encoder architecture. Key innovations include:

- **Multi-Patch Size Projection Layers**: Handle heterogeneous frequencies by mapping them to specialized patch sizes
- **Any-variate Attention**: Flattens multivariate data into a single sequence with binary attention bias
- **Mixture Distribution Head**: Outputs probabilistic forecasts using Student's t, Negative Binomial, Log-normal, and Normal distributions
- **Sequence Packing**: Reduces padding waste from ~61% to <0.4%

### 1.2 Important Architecture Discovery

**MOIRAI Already Has Per-Variate Normalization**: The current `PackedStdScaler` (`src/uni2ts/module/packed_scaler.py`) already computes mean and standard deviation **independently for each variate** via:
```python
id_mask = torch.logical_and(
    torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
    torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
)
```

This means Open, High, Low, Close, Volume each get their own `loc` and `scale` values **automatically**. Volume normalization using the existing scaler is already supported.

**What We Need to Add**: Collective OHLC normalization (one mean/std across all 4 price columns) requires a **new custom scaler**.

### 1.3 Current Limitations for Financial Data

MOIRAI's design has limitations for OHLCV data:
- **Binary Attention Bias**: Currently only distinguishes "same variate" vs "different variate" - doesn't understand semantic relationships
- **No Type Awareness**: Doesn't know that Open, High, Low, Close are semantically related price columns
- **No Target Specification**: Treats all variates equally in prediction (we want to focus on Close)
- **Missing Directional Loss**: Standard NLL loss doesn't directly optimize for directional accuracy

---

## 2. Data Structure and Requirements

### 2.1 Confirmed Data Schema

**Parquet Files Structure**:
```
ts                    open      high      low       close     volume
2000-01-03 14:30:00  56.3305   56.3305   56.3305   56.3305   146510.0
2000-01-03 14:35:00  56.3305   56.4646   55.7940   56.1069   98559.0
...
```

**Key Features**:
- Datetime index: `ts` column
- Price columns: `open`, `high`, `low`, `close`
- Volume column: `volume`
- Resolution: 5-minute intervals (primary)
- Available resolutions: 1d, 1h, 5m (in `data/processed_equities/`)

### 2.2 Prediction Task Definition

**Setup**:
- **Input**: Multivariate time series with 5 variates [open, high, low, close, volume]
- **Context Length**: To be determined (e.g., 512 time steps ≈ 2 days)
- **Prediction Length**: To be determined (e.g., 12 time steps = 1 hour)
- **Target**: Close price only (but model predicts all 5 during training)
- **Use Case**: Predict Close price using OHLCV context to inform potential price changes

**Important Note**: During training, MOIRAI predicts ALL 5 variates. We don't change this - we simply evaluate focus on the Close price predictions. The other 4 variates (OHLC + volume) serve as context but are still predicted by the model during the masked training objective.

### 2.3 Normalization Requirements

**Volume Column**:
- Use existing `PackedStdScaler` (per-variate Z-score normalization)
- Already supported by MOIRAI
- No changes needed

**OHLC Columns**:
- **Require new custom scaler**: `CollectiveOHLCScaler`
- Single mean and standard deviation computed across all 4 price columns
- All OHLC values normalized together as they represent the same asset's price
- Example: If mean=[50, 51, 49, 50.5] and std=[5, 5.2, 4.8, 5.1], we compute global_mean=50.1 and global_std=5.0
- Normalized values: `(value - global_mean) / global_std`

---

## 3. Core Innovation: Semantic Variate ID System

### 3.1 ID Allocation Scheme

We introduce an integer ID system to uniquely identify each variate type:

| ID | Variate Type | Description |
|----|--------------|-------------|
| 0 | Anonymous | Default/unknown variate (backward compatibility) |
| 1 | Close | Closing price (primary prediction target) |
| 2 | Open | Opening price |
| 3 | High | Highest price |
| 4 | Low | Lowest price |
| 5 | Volume | Trading volume |
| 6-255 | Reserved | Reserved for future core indicators |
| 256+ | Custom | User-defined technical indicators |

**Design Rationale:**
- **Integers**: Memory-efficient compared to full embeddings
- **Reserved space**: IDs 6-255 reserved for future standard indicators (RSI, MACD, etc.)
- **Custom range**: Large space for user-defined indicators without conflicts
- **Backward compatibility**: ID 0 maintains compatibility with existing MOIRAI models

### 3.2 Implementation Architecture

```python
class VariateIDRegistry:
    """
    Registry for managing variate type IDs.
    Ensures consistent ID allocation across training and inference.
    """
    
    def __init__(self):
        # Core OHLCV IDs
        self.core_ids = {
            'anonymous': 0,
            'close': 1,
            'open': 2,
            'high': 3,
            'low': 4,
            'volume': 5
        }
        
        # Reserved for future standard indicators
        self.reserved_range = (6, 255)
        
        # Custom indicator range
        self.custom_range = (256, 65535)
        
        # Track allocated IDs
        self.allocated_ids = set(self.core_ids.values())
        self.next_custom_id = 256
    
    def register_indicator(self, name: str) -> int:
        """Register a new custom indicator and return its ID."""
        if name in self.core_ids:
            return self.core_ids[name]
        
        # Check if already registered
        if hasattr(self, 'custom_ids') and name in self.custom_ids:
            return self.custom_ids[name]
        
        # Allocate new ID
        if self.next_custom_id > 65535:
            raise ValueError("Exhausted variate ID space")
        
        var_id = self.next_custom_id
        self.next_custom_id += 1
        
        if not hasattr(self, 'custom_ids'):
            self.custom_ids = {}
        self.custom_ids[name] = var_id
        self.allocated_ids.add(var_id)
        
        return var_id
    
    def get_id(self, name: str) -> int:
        """Get ID for a variate type."""
        if name in self.core_ids:
            return self.core_ids[name]
        if hasattr(self, 'custom_ids') and name in self.custom_ids:
            return self.custom_ids[name]
        raise ValueError(f"Unknown variate type: {name}")
```

### 3.3 Adding Variate Type IDs to Data Pipeline

**Current State**: `AddVariateIndex` in `src/uni2ts/transform/feature.py` creates `variate_id` field which identifies which variate (0, 1, 2, 3, 4) each time step belongs to.

**What We Need**: Add a NEW transformation `AddVariateTypeID` that assigns semantic type IDs (close=1, open=2, high=3, low=4, volume=5) to each variate.

```python
@dataclass
class AddVariateTypeID(Transformation):
    """
    Add semantic variate type IDs to data_entry.
    Maps column names to semantic types (close=1, open=2, high=3, low=4, volume=5).
    """
    
    variate_type_field: str = "variate_type_id"
    
    # Mapping from column names to type IDs
    column_to_type_id = {
        'close': 1,
        'open': 2,
        'high': 3,
        'low': 4,
        'volume': 5
    }
    
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """
        Add variate_type_id field indicating semantic type of each variate.
        
        Args:
            data_entry: Dictionary containing time series data
        
        Returns:
            Updated data_entry with variate_type_id field
        """
        # Assuming data_entry['target'] is shape [num_variates, seq_len]
        # We need to create variate_type_id of same shape
        num_variates = len(data_entry['target'])
        
        # Map each variate index to its type ID
        # This assumes column order is fixed: [close, open, high, low, volume]
        # or we can use column names if available
        
        type_ids = []
        for i in range(num_variates):
            # Get column name or use default mapping
            col_name = self._get_column_name(i, data_entry)
            type_id = self.column_to_type_id.get(col_name, 0)  # Default to anonymous
            type_ids.append(type_id)
        
        # Create tensor of shape [num_variates, seq_len]
        seq_len = data_entry['target'].shape[1]
        var_type_id_tensor = np.array(type_ids).reshape(-1, 1)
        var_type_id_tensor = np.tile(var_type_id_tensor, (1, seq_len))
        
        data_entry[self.variate_type_field] = var_type_id_tensor
        return data_entry
    
    def _get_column_name(self, var_idx: int, data_entry: dict) -> str:
        """Get column name from variate index."""
        # Implementation depends on data format
        # May need to pass column names via metadata
        column_names = ['close', 'open', 'high', 'low', 'volume']
        return column_names[var_idx] if var_idx < len(column_names) else 'unknown'
```

### 3.4 Benefits of the ID System

1. **Extensibility**: Add new indicators without modifying core architecture
2. **Efficiency**: Integer IDs are memory-efficient
3. **Semantic Awareness**: Model can learn relationships between specific variate types
4. **Backward Compatibility**: ID 0 maintains compatibility with existing models
5. **Scalability**: Sufficient IDs for most use cases
6. **Non-destructive**: New indicators don't require retraining existing embeddings

---

## 4. Architecture Modifications

### 4.1 Custom Normalization: Collective OHLC Scaler

**CRITICAL**: MOIRAI's existing `PackedStdScaler` normalizes each variate separately. We need a custom scaler that normalizes OHLC collectively.

```python
class CollectiveOHLCScaler(nn.Module):
    """
    Normalize OHLC columns collectively using single mean and std.
    Volume columns are normalized individually.
    """
    
    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale
    
    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        var_type_id: Int[torch.Tensor, "*batch seq_len"],  # New: semantic type IDs
    ) -> tuple[
        Float[torch.Tensor, "*batch 1 max_patch"],
        Float[torch.Tensor, "*batch 1 max_patch"]
    ]:
        """
        Compute collective OHLC normalization statistics.
        
        Args:
            target: Input data [..., seq_len, num_variates]
            observed_mask: Mask for observed values
            sample_id: Sample indices for packing
            variate_id: Variate indices (0,1,2,3,4)
            var_type_id: Semantic type IDs (1=close, 2=open, 3=high, 4=low, 5=volume)
        
        Returns:
            loc: Mean values for each variate
            scale: Standard deviation values for each variate
        """
        batch_shape = target.shape[:-2]
        num_variates = target.shape[-1]
        
        # Initialize loc and scale
        loc = torch.zeros(batch_shape + (1, num_variates), 
                         dtype=torch.float64, device=target.device)
        scale = torch.ones(batch_shape + (1, num_variates), 
                          dtype=torch.float64, device=target.device)
        
        # Get collective OHLC mask (types 1-4 are OHLC, type 5 is volume)
        ohlc_mask = ((var_type_id >= 1) & (var_type_id <= 4))
        volume_mask = (var_type_id == 5)
        
        # Compute collective OHLC statistics
        if ohlc_mask.any():
            # Extract OHLC values
            ohlc_values = target * ohlc