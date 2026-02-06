# Transform Module

## Overview

The transform module provides a comprehensive data transformation pipeline for time series data in MOIRAI. It includes transformations for patching, masking, normalization, feature engineering, and sequence manipulation. All transformations follow a composable design pattern using the Chain class and Transformation interface.

## Core Architecture (`_base.py`)

### Transformation Interface

**Transformation Base Class:** Abstract base defining the transformation contract.

**Key Methods:**
- `__call__()`: Transform a data entry dictionary
- `chain()`: Combine with another transformation
- `__add__()`: Operator overloading for chaining

**Composition Operators:**
```python
transform = (
    AddTimeIndex()
    .chain(AddObservedMask())
    .chain(Patch(patch_size=32))
    .chain(Pack(scaler="std"))
)
# Equivalent:
transform = AddTimeIndex() + AddObservedMask() + Patch(patch_size=32) + Pack(scaler="std")
```

### Chain Class

**Functionality:**
- Sequential application of transformations
- Automatic flattening of nested chains
- Optimization by removing Identity transforms

### Mixins (`_mixin.py`)

**MapFuncMixin:** Applies transformations to multiple fields
**CollectFuncMixin:** Collects results from multiple fields
**CheckArrNDimMixin:** Validates array dimensions

## Patching System (`patch.py`)

### Patch Size Constraints

**FixedPatchSizeConstraints:** Explicit min/max patch sizes
**DefaultPatchSizeConstraints:** Frequency-based ranges

**Frequency Guidelines:**
- Seconds (S): 64-128 (8min - 68min patches)
- Minutes (T): 32-128 (1h - 8.5h patches)  
- Hours (H): 32-64 (5.3 days patches)
- Days (D/B): 16-32 patches
- Weeks (W): 16-32 patches
- Months (M): 8-32 patches
- Quarters/Years (Q/Y/A): 1-8 patches

### GetPatchSize Transformation

**Purpose:** Dynamically select appropriate patch size based on:
- Data frequency constraints
- Minimum time patches requirement
- Sequence length limitations

**Algorithm:**
1. Calculate maximum patch size based on `min_time_patches`
2. Filter candidates by frequency constraints and length
3. Randomly sample from valid patch sizes

### Patchify Transformation

**Purpose:** Divide time series into fixed-size patches for transformer input.

**Process:**
1. Split sequences into patches along time dimension
2. Pad to `max_patch_size` with specified value
3. Rearrange from `[... time*patch]` to `[... time max_patch]`

## Task-Specific Transformations (`task.py`)

### MaskedPrediction

**Purpose:** Random masking for BERT-style pre-training.

**Features:**
- Configurable masking ratio range (`min_mask_ratio` to `max_mask_ratio`)
- Masks trailing portion of sequences (prediction horizon)
- Truncates other fields to prevent data leakage
- Generates `prediction_mask` for loss computation

### EvalMaskedPrediction

**Purpose:** Fixed-length masking for evaluation and fine-tuning.

**Features:**
- Fixed `mask_length` for consistent evaluation
- Masks last N time steps as prediction horizon
- Truncates input sequences accordingly

## Sequence Manipulation

### Crop/Pad (`crop.py`, `pad.py`)

**Crop:** Truncate sequences to maximum length
**Pad:** Extend sequences with padding values

### Resample (`resample.py`)

**Purpose:** Change temporal resolution of time series.

**Types:**
- Upsampling (increase frequency)
- Downsampling (decrease frequency)
- Aggregation methods (mean, sum, etc.)

### Reshape (`reshape.py`)

**Purpose:** Restructure array shapes between transformations.

**Operations:**
- Transpose dimensions
- Join/split along axes
- Reshape for different representation formats

## Feature Engineering (`feature.py`)

**AddTimeIndex:** Convert pandas timestamps to integer indices
**AddObservedMask:** Create masks for valid observations
**AddField:** Add static or computed features

## Normalization (`field.py`)

**Scalers Integration:** Apply various scaling transformations per field.

**Supported Scalers:**
- StandardScaler (z-score)
- MinMaxScaler
- RobustScaler
- Packed versions for sequence-aware scaling

## Time Series Operations

### Imputation (`imputation.py`)

**Methods:**
- Forward fill/backward fill
- Linear interpolation
- Constant value imputation
- Statistical imputation (mean, median)

## Relationship to Other Modules

### With Data Module
- Transformations applied in dataset pipelines
- Used by DataLoaders for batch processing
- Integrates with dataset split operations

### With Model Module
- Transforms data to model-expected format
- Patching provides Transformer input shape
- Masking defines loss computation regions

### With Loss Module
- `prediction_mask` controls loss calculation
- Supports different masking strategies for training

## Design Patterns

### Functional Composition
```python
# Declarative pipeline definition
transform = (
    AddTimeIndex()
    + AddObservedMask() 
    + GetPatchSize(min_time_patches=10)
    + Patchify(max_patch_size=128)
    + MaskedPrediction(min_mask_ratio=0.1, max_mask_ratio=0.5)
)
```

### Type Safety
- Extensively uses `jaxtyping` annotations
- Validates array shapes at runtime
- Prevents dimension mismatches

### Mixin Pattern
- Composable functionality through multiple inheritance
- Specialized mixins for common operations
- Avoids code duplication

### Configuration-Driven
- Most transforms accept parameters for flexibility
- Supports optional fields with graceful handling
- Backward compatibility through optional arguments

## Critical Path Examples

### Pre-training Pipeline
```python
pretrain_transform = (
    AddTimeIndex()
    + AddObservedMask()
    + GetPatchSize(min_time_patches=10)
    + Patchify(max_patch_size=128)
    + MaskedPrediction(min_mask_ratio=0.1, max_mask_ratio=0.5)
    + Pack(scaler=PackedStdScaler())
)
```

### Fine-tuning Pipeline
```python
finetune_transform = (
    AddTimeIndex()
    + AddObservedMask()
    + GetPatchSize(min_time_patches=10)
    + EvalMaskedPrediction(mask_length=24)
    + Pack(scaler=PackedStdScaler())
)
```

### Evaluation Pipeline
```python
eval_transform = (
    AddTimeIndex()
    + AddObservedMask()
    + GetPatchSize(min_time_patches=10)  
    + EvalMaskedPrediction(mask_length=prediction_length)
    + Pack(scaler=PackedStdScaler())
)
```

## Advanced Features

### Multi-Frequency Support
- Automatic patch size selection based on frequency
- Time-aware constraints for different granularities
- Frequency-specific default ranges

### Sequence Packing Ready
- Compatible with MOIRAI's packed batching strategy
- Maintained sample/variate ID grouping
- Optimized for variable-length sequences

### Feature Flexibility
- Apply transforms to multiple fields simultaneously
- Skip optional fields if missing
- Different handling for different data types (univariate vs multivariate)

This module provides the data preprocessing foundation that enables MOIRAI's universal time series forecasting capabilities across diverse domains and frequencies.
