# Optim Module

## Overview

The optim module provides learning rate schedulers and optimization utilities for MOIRAI model training. It includes a comprehensive collection of scheduler types optimized for transformer training, with support for warm-up phases and various decay patterns.

## Core Components (`lr_scheduler.py`)

### Scheduler Functions

All schedulers return PyTorch `LambdaLR` or `ReduceLROnPlateau` objects that can be used directly with any optimizer.

#### Basic Schedulers

**Constant Schedule** `get_constant_schedule()`
- Maintains constant learning rate throughout training
- No decay or warmup phases

**Reduce on Plateau** `get_reduce_on_plateau_schedule()`
- Reduces learning rate when validation metric stops improving
- Uses PyTorch's `ReduceLROnPlateau` scheduler

#### Warmup Schedulers

**Constant with Warmup** `get_constant_schedule_with_warmup()`
- Linear increase from 0 to initial LR during warmup
- Constant LR after warmup period
- Used for stable training convergence

**Inverse Square Root** `get_inverse_sqrt_schedule()`
- Inverse square root decay after warmup
- Similar to transformer training schedules
- Mathematically: LR = 1/sqrt(t)

#### Decay Schedulers

**Linear Schedule with Warmup** `get_linear_schedule_with_warmup()`
- Linear increase during warmup (0 to initial LR)
- Linear decrease after warmup (initial LR to 0)
- Common in fine-tuning scenarios

**Cosine Schedule with Warmup** `get_cosine_schedule_with_warmup()`
- Linear warmup followed by cosine decay
- Smooth decay curve avoiding sharp drops
- Can be configured with multiple cycles

**Cosine with Hard Restarts** `get_cosine_with_hard_restarts_schedule_with_warmup()`
- Cosine decay with periodic restarts
- Useful for preventing local minima
- Number of cycles configurable

**Polynomial Decay with Warmup** `get_polynomial_decay_schedule_with_warmup()`
- Power-law decay after warmup
- Configurable end learning rate
- Default power=1.0 (linear decay)

### Unified API

**get_scheduler() Function**
```python
def get_scheduler(
    name: str | SchedulerType,
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    scheduler_specific_kwargs: Optional[dict] = None,
):
```

**Features:**
- String or enum-based scheduler selection
- Unified interface across all scheduler types
- Automatic parameter validation
- Support for scheduler-specific configurations

**Supported Schedulers:**
```python
class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine" 
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"
```

## Usage Patterns

### Basic Usage
```python
from uni2ts.optim import get_scheduler
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000
)

# Use in training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

### Lightning Integration
```python
class MoiraiModule(pl.LightningModule):
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        scheduler = get_scheduler(
            "linear",
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
```

### Reduce on Plateau
```python
scheduler = get_scheduler(
    "reduce_lr_on_plateau",
    optimizer,
    scheduler_specific_kwargs={
        "mode": "min",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-7
    }
)
```

## Design Philosophy

### Warmup Importance
- Prevents catastrophic forgetting in fine-tuning
- Stabilizes training at the beginning
- Gradually introduces higher learning rates

### Decay Strategies
- **Linear**: Simple, predictable decay
- **Cosine**: Smooth, theoretically motivated
- **Polynomial**: Configurable decay rates
- **Inverse sqrt**: Transformer standard

### Scheduler Selection Guidelines

**For Pre-training:**
- Use cosine with warmup for long training runs
- Inverse square root for large datasets

**For Fine-tuning:**
- Linear with warmup for most tasks
- Constant with warmup for stable domains

**For Validation-based Training:**
- Reduce on plateau for early stopping scenarios

## Implementation Details

### Mathematical Formulas

**Cosine with Warmup:**
```python
if step < num_warmup_steps:
    lr = (step / num_warmup_steps) * initial_lr
else:
    progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    lr = 0.5 * initial_lr * (1 + cos(π * num_cycles * 2 * progress))
```

**Linear with Warmup:**
```python
if step < num_warmup_steps:
    lr = (step / num_warmup_steps) * initial_lr
else:
    progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    lr = initial_lr * (1 - progress)
```

### Parameter Validation
- Validates required parameters for each scheduler type
- Prevents invalid combinations (e.g., warmup without steps)
- Clear error messages for configuration issues

### Backward Compatibility
- Based on Hugging Face transformers scheduler implementations
- Compatible with existing training pipelines
- Clean migration path from other frameworks

This module provides the learning rate scheduling foundation for effective MOIRAI model training across diverse scenarios and datasets.