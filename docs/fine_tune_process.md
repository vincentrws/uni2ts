# MOIRAI Fine-Tuning Process: Complete Walkthrough

This document details the accurate flow of the fine-tuning process for dataset A, which uses the standard HuggingFace Dataset + FinetunePatchCrop pipeline.

## 1. DATA PREPARATION PIPELINE

### Step 1.1: Raw Data → CSV Conversion (`src/uni2ts/data/builder/turn_parquet_csv.py`)

Your parquet files are converted to CSV with proper time features:

```
Input: AAPL.parquet [ts, open, high, low, close, volume]
  ↓
1. Load & set 'ts' as DatetimeIndex (UTC)
2. Resample to 5min (fills gaps with NaN)
3. Calculate time features:
   - Minutes after market open (0-390)
   - Day of week (0-4)
4. Detect trading hours from observed data
  ↓
Output: A.csv
[ts, Open, High, Low, Close, Volume, Minutes after market open, DayOfWeek]
```
**Key:** NaN values mark market gaps (weekends, holidays).

### Step 1.2: CSV → HuggingFace Dataset (`project/moirai-1/finetune_lsf/build_A_dataset.sh`)

```bash
python -m uni2ts.data.builder.simple \
  A \
  "data/csv/A.csv" \
  --dataset_type "wide_multivariate" \
  --freq "5min"
```

Calls `src/uni2ts/data/builder/simple.py:_from_wide_dataframe_multivariate()`:

```
Input: A.csv (7 columns)
  ↓
For each row (timestamp):
  - Create sample with all 7 variates
  - target = [Open, High, Low, Close, Volume, Minutes, DayOfWeek]
  - start = timestamp
  - freq = "5min"
  - item_id = "A"
  ↓
Output: HuggingFace Dataset saved to disk
  {
    "item_id": "A",
    "start": Timestamp,
    "freq": "5min",
    "target": [7 values per timestep]
  }
```

## 2. FINETUNE DATASET BUILDER (`src/uni2ts/data/builder/simple.py:SimpleFinetuneDatasetBuilder`)

### Step 2.1: Load HuggingFace Dataset (`src/uni2ts/data/builder/simple.py:319-341`)

```python
def load_dataset(self, transform_map):
    # Load saved HuggingFace dataset
    hf_dataset = datasets.load_from_disk(
        "custom_data/lsf/wide_multivariate/A"
    )
    
    # Wrap in HuggingFaceDatasetIndexer
    indexer = HuggingFaceDatasetIndexer(hf_dataset)
    
    # Create FinetuneDataset with FinetunePatchCrop transform
    return FinetuneDataset(
        windows=windows,  # Number of sliding windows
        indexer=indexer,
        transform=transform_map["A"](
            distance=distance,
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=patch_size,
        ),
    )
```
**Key:** `windows` parameter controls how many training windows are created.

## 3. FINETUNE DATASET: WINDOWING (`src/uni2ts/data/dataset.py:FinetuneDataset`)

### Step 3.1: Calculate Windows

From training script:
```bash
train_length=8640          # Total timesteps available
context_length=512         # Input window
prediction_length=96       # Forecast horizon
distance=1                 # Stride between windows
```

Windows calculation (`src/uni2ts/data/builder/simple.py:438`):
```python
windows = (train_length - context_length - prediction_length) // distance + 1
        = (8640 - 512 - 96) // 1 + 1
        = 8032 + 1
        = 8033 windows
```

Each window contains:
- **Context:** 512 timesteps (input to model)
- **Prediction:** 96 timesteps (target for model)
- **Total:** 608 timesteps per window

### Step 3.2: FinetuneDataset.getitem() (`src/uni2ts/data/dataset.py:235-239`)

```python
def _get_data(self, idx: int) -> dict[str, Data]:
    window, idx = divmod(idx, self.num_ts)
    item = self.indexer[idx]  # Load from HuggingFace dataset
    item["window"] = window    # Add window index
    return item
```

**Example:** If `idx=5000`:
- `window = 5000 // 1 = 5000`
- `idx = 5000 % 1 = 0`
- Load item 0 (stock "A") with window=5000

## 4. FINETUNE PATCH CROP TRANSFORM (`src/uni2ts/transform/crop.py:FinetunePatchCrop`)

This is the critical windowing step that extracts context + prediction windows.

### Step 4.1: Calculate Boundaries (`src/uni2ts/transform/crop.py:177-187`)

```python
def _get_boundaries(self, data_entry):
    time = data_entry["target"][0].shape[0]  # Total timesteps (8640)
    window = data_entry["window"]             # Window index (e.g., 5000)
    
    # Calculate where forecast starts
    fcst_start = context_length + window * distance
                = 512 + 5000 * 1
                = 5512
    
    # Calculate crop boundaries
    a = fcst_start - context_length
      = 5512 - 512
      = 5000
    
    b = fcst_start + prediction_length
      = 5512 + 96
      = 5608
    
    return a, b  # Crop indices [5000:5608]
```

### Step 4.2: Crop Data (`src/uni2ts/transform/crop.py:174-175`)

```python
def _crop(data_entry, field, a, b):
    return [ts[a:b] for ts in data_entry[field]]
```

**For field "target":**
```
Original: [8640 timesteps, 7 variates]
Cropped:  [608 timesteps, 7 variates]
          = [512 context + 96 prediction, 7 variates]
```

**For field "past_feat_dynamic_real":**
```
Original: [6, 8640]  (6 features, 8640 timesteps)
Cropped:  [6, 608]   (6 features, 608 timesteps)
```

## 5. BATCH COLLATION (`src/uni2ts/data/loader.py:PadCollate`)

### Step 5.1: Pad Samples to Max Length

```python
def pad_samples(self, batch):
    for sample in batch:
        length = len(sample["target"])  # 608
        
        # Pad to max_length (usually context_length)
        for key in seq_fields:
            sample[key] = torch.cat([
                sample[key],  # [608, 7]
                pad_func(
                    (max_length - length,) + sample[key].shape[1:],
                    sample[key].dtype
                )  # Padding if needed
            ])
```

### Step 5.2: Create Batch

```python
def __call__(self, batch):
    sample_id = self.get_sample_id(batch)  # [B, max_length]
    padded_batch = self.pad_samples(batch)
    
    return {
        "target": torch.stack([s["target"] for s in batch]),
        "past_feat_dynamic_real": torch.stack([...]),
        "sample_id": sample_id,
        "variate_id": torch.arange(7).repeat(max_length, 1),
        ...
    }
```

**Output batch shape:**
```
{
  "target": [B, 608, 7],                    # B batches, 608 timesteps, 7 variates
  "past_feat_dynamic_real": [B, 6, 608],    # 6 features (Open, High, Low, Vol, Min, Dow)
  "sample_id": [B, 608],                    # Batch index per timestep
  "variate_id": [B, 608, 7],                # Variate index per position
  "observed_mask": [B, 608, 7],             # Valid data mask (from NaN detection)
  ...
}
```

## 6. OHLCV SCALER: NORMALIZATION (`src/uni2ts/module/packed_scaler.py:OHLCVPackedScaler`)

The scaler is applied during model forward pass to normalize the batch.

### Step 6.1: Variate Grouping

```
Variate indices in packed data:
  0: Open   → Group 0 (OHLC collective)
  1: High   → Group 0 (OHLC collective)
  2: Low    → Group 0 (OHLC collective)
  3: Close  → Group 0 (OHLC collective)
  4: Volume → Group 1 (individual)
  5: Minutes→ Group 2 (individual)
  6: DayOfWeek → Group 3 (individual)
```

### Step 6.2: Compute Statistics Per Group (`src/uni2ts/module/packed_scaler.py:607-641`)

**For Group 0 (OHLC):**

Input batch: `[B, 608, 7]`

For each sample in batch:
  - Collect all OHLC values (indices 0,1,2,3)
  - Compute mean across all 4 variates
  - Compute std across all 4 variates
  - Apply same (mean, std) to all 4 variates

**Example:**
```
OHLC values in window: [100, 105, 98, 103, 102, 104, 99, 101, ...]
mean_ohlc = 101.5
std_ohlc = 2.3

Open_norm = (100 - 101.5) / 2.3 = -0.65
High_norm = (105 - 101.5) / 2.3 = 1.52
Low_norm = (98 - 101.5) / 2.3 = -1.52
Close_norm = (103 - 101.5) / 2.3 = 0.65
```
**Why collective?** OHLC prices share the same unit (dollars). Collective scaling preserves relationships: High ≥ Close ≥ Open ≥ Low.

### Step 6.3: Output Normalized Batch

```python
loc, scale = scaler(target, observed_mask, sample_id, variate_id)
normalized_target = (target - loc) / scale

# Output: [B, 608, 7] normalized values
```

## 7. MODEL FORWARD PASS (`src/uni2ts/model/moirai/module.py`)

Input: `normalized_target [B, 608, 7]`
  ↓
1. **Patch Embedding**
   - Split 608 timesteps into patches of size 64
   - 608 / 64 = 9.5 → 9 full patches + 1 partial
   - Multi-scale: [8, 16, 32, 64, 128] patch sizes
  ↓
2. **Transformer Encoder** (6 layers)
   - d_model = 384
   - Attention with binary variate bias
   - Output: [B, 608, 384]
  ↓
3. **Distribution Head**
   - Mixture of 4 distributions (StudentT, NormalFixedScale, NegativeBinomial, LogNormal)
   - Output: distribution parameters for each timestep
  ↓
**Output:** Predicted distribution parameters `[B, 608, ...]`

## 8. LOSS & OPTIMIZATION

### Step 8.1: Loss Computation (`cli/conf/finetune/model/moirai_1.1_R_small_ohlcv.yaml:27-28`)

```python
loss = PackedNLLLoss(
    predictions,           # Distribution parameters
    normalized_target,     # [B, 608, 7]
    observed_mask          # [B, 608, 7] - ignores gaps
)
```
**Key:** `observed_mask` ensures NaN positions (market gaps) don't contribute to loss.

### Step 8.2: Backward Pass & Optimization

```python
loss.backward()
optimizer.step()  # Adam with lr=5e-7
```

## 9. COMPLETE DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. RAW DATA                                                     │
│    AAPL.parquet: [ts, open, high, low, close, volume]          │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. CSV CONVERSION (turn_parquet_csv.py)                         │
│    A.csv: [ts, Open, High, Low, Close, Volume, Minutes, DayOfWeek]
│    - Resample to 5min, fill gaps with NaN                       │
│    - Calculate time features                                    │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. DATASET BUILDER (simple.py)                                  │
│    HuggingFace Dataset: {item_id, start, freq, target: [7]}    │
│    Saved to: custom_data/lsf/wide_multivariate/A               │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. FINETUNE DATASET (dataset.py:FinetuneDataset)                │
│    - Load HuggingFace dataset                                   │
│    - Create 8033 windows (train_length - context - pred) / dist │
│    - Each window: [context_length + prediction_length]          │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. FINETUNE PATCH CROP (crop.py:FinetunePatchCrop)              │
│    - Calculate boundaries: a = context + window*distance        │
│    - Crop: [a:a+context+prediction]                             │
│    - Output: [608, 7] (512 context + 96 prediction)             │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. BATCH COLLATION (loader.py:PadCollate)                       │
│    - Stack B samples                                            │
│    - Pad to max_length if needed                                │
│    - Create sample_id, variate_id, observed_mask                │
│    - Output: [B, 608, 7] batch                                  │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. OHLCV SCALER (packed_scaler.py:OHLCVPackedScaler)            │
│    - Group OHLC (0,1,2,3) → collective z-score                  │
│    - Group Volume (4) → individual z-score                      │
│    - Group Time features (5,6) → individual z-score             │
│    - Normalize: (value - mean) / std                            │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. MODEL FORWARD PASS (module.py:MoiraiModule)                  │
│    - Patch embedding (multi-scale)                              │
│    - Transformer encoder (6 layers, d_model=384)                │
│    - Distribution head (mixture of 4 distributions)             │
│    - Output: distribution parameters                            │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ 9. LOSS & OPTIMIZATION                                          │
│    - PackedNLLLoss (respects observed_mask)                     │
│    - Backward pass                                              │
│    - Adam optimizer (lr=5e-7, weight_decay=1e-1)                │
└─────────────────────────────────────────────────────────────────┘
```

## 10. KEY CONFIGURATION PARAMETERS

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `train_length` | 8640 | Total available timesteps (8640 × 5min = 30 days) |
| `context_length` | 512 | Input window (512 × 5min = 42.7 hours) |
| `prediction_length` | 96 | Forecast horizon (96 × 5min = 8 hours) |
| `distance` | 1 | Stride between windows (1 = overlapping) |
| `windows` | 8033 | Number of training windows |
| `patch_size` | 64 | Patch size (512/64 = 8 patches) |
| `batch_size` | Varies | Samples per batch |
| `max_epochs` | 50 | Training epochs |
| `lr` | 5e-7 | Learning rate |

## 11. DATA PREPARATION SUMMARY

Your data preparation pipeline:

1. **Parquet → CSV**: Resample to 5min, fill gaps with NaN, add time features.
2. **CSV → HuggingFace**: Create multivariate dataset with 7 variates per timestep.
3. **HuggingFace → Windows**: `FinetunePatchCrop` creates sliding windows of [context + prediction].
4. **Batch Creation**: `PadCollate` stacks windows into batches with proper masking.
5. **Normalization**: `OHLCVPackedScaler` applies collective OHLC scaling + individual scaling for other features.
6. **Model Training**: Transformer processes normalized batches and predicts distributions.

The key innovation is **collective OHLC scaling** which preserves price relationships while normalizing for training.
