# Technical Report: MOIRAI Time Series Foundation Model

## Table of Contents
- [Technical Report: MOIRAI Time Series Foundation Model](#technical-report-moirai-time-series-foundation-model)
  - [Table of Contents](#table-of-contents)
  - [1. Executive Summary](#1-executive-summary)
  - [2. Core Architectural Innovations](#2-core-architectural-innovations)
    - [2.1 Multi-Patch Size Projection Layers](#21-multi-patch-size-projection-layers)
    - [2.2 Any-variate Attention](#22-any-variate-attention)
    - [2.3 Mixture Distribution Head](#23-mixture-distribution-head)
  - [3. The LOTSA Dataset](#3-the-lotsa-dataset)
  - [4. Unified Training Methodology](#4-unified-training-methodology)
  - [5. Performance Evaluation](#5-performance-evaluation)
    - [5.1 In-distribution (Monash Benchmark)](#51-in-distribution-monash-benchmark)
    - [5.2 Zero-Shot Forecasting (Out-of-Distribution)](#52-zero-shot-forecasting-out-of-distribution)
  - [6. Key Insights and Ablations](#6-key-insights-and-ablations)
  - [7. Model Variants](#7-model-variants)
  - [8. Conclusion](#8-conclusion)

---

## 1. Executive Summary

The **MOIRAI** (Masked Encoder-based UnIveRsAl TIme Series Forecasting Transformer) represents a shift in time series forecasting from specialized, dataset-specific models to **Universal Time Series Models (UTSMs)**. Developed by Salesforce AI Research, MOIRAI is a foundation model pre-trained on the LOTSA archive (27B observations), designed to handle any frequency, any number of variates, and diverse distributional properties in a zero-shot capacity.

---

## 2. Core Architectural Innovations

MOIRAI addresses three fundamental challenges in universal forecasting: cross-frequency learning, any-variate dimensionality, and varying distributions.

### 2.1 Multi-Patch Size Projection Layers

Traditional Transformers use a fixed patch size. MOIRAI introduces multiple input and output projection layers to handle heterogeneous frequencies.

**Mechanism:** It maps specific frequencies to specialized patch sizes. Larger patches are used for high-frequency data (e.g., seconds/minutes) to reduce the quadratic cost of attention, while smaller patches are used for low-frequency data (e.g., yearly/monthly) to maximize semantic extraction.

**Standard Mapping:**
- Yearly/Quarterly: Patch size 8
- Monthly: Patch sizes 8, 16, 32
- Weekly/Daily: Patch sizes 16, 32
- Hourly: Patch sizes 32, 64
- Minute/Second-level: Patch sizes 64, 128

### 2.2 Any-variate Attention

To handle multivariate data with an arbitrary number of variables, MOIRAI "flattens" the variates into a single sequence.

**Temporal Encoding:** Uses Rotary Position Embeddings (RoPE) to encode the time index \(i-j\), ensuring the model understands the relative temporal distance between patches.

**Variate Encoding:** Instead of fixed embeddings, it uses Binary Attention Bias. This adds a learnable scalar \(u^{(1)}\) if two patches belong to the same variate (\(m=n\)) and \(u^{(2)}\) if they belong to different variates (\(m \neq n\)).

**Equivariance:** This design ensures the model is invariant to the order of variates and can scale to an unlimited number of sensors or exogenous variables.

### 2.3 Mixture Distribution Head

Standard forecasting models often assume a single distribution (e.g., Gaussian). MOIRAI outputs a mixture of parametric distributions to handle varied data types:
- **Student's t-distribution:** For general, robust forecasting.
- **Negative Binomial:** Specifically for positive count data (e.g., sales).
- **Log-normal:** For right-skewed phenomena (e.g., economic data).
- **Low-variance Normal:** For high-confidence, stable predictions.

---

## 3. The LOTSA Dataset

A critical component of MOIRAI's success is the **Large-scale Open Time Series Archive (LOTSA)**.

- **Scale:** 27.6 Billion observations (reaching 231B if counting across all variates).
- **Domains:** Nine distinct domains including Energy (59%), Transport (17.7%), Climate (15%), CloudOps, Web, Sales, Nature, Finance, and Healthcare.
- **Diversity:** Spans frequencies from yearly to second-level.

---

## 4. Unified Training Methodology

MOIRAI is trained using a **Masked Encoder** objective, similar to BERT or masked autoencoders in vision.

- **Task Distribution:** Unlike models trained for a fixed \(L\) (lookback) and \(H\) (horizon), MOIRAI samples \(L\) and \(H\) dynamically during training. The total sequence length is capped at 512 tokens.
- **Sequence Packing:** To optimize throughput, multiple short time series are "packed" into a single training sequence, reducing padding from ~61% to less than 0.4%.
- **Objective:** Minimization of the Negative Log-Likelihood (NLL) of the mixture distribution.

---

## 5. Performance Evaluation

### 5.1 In-distribution (Monash Benchmark)

MOIRAI was tested on the held-out test sets of the Monash archive.

**Results:** MOIRAI (Base and Large) outperformed all traditional baselines (ARIMA, CatBoost, DeepAR) as a single model, whereas baselines required individual training for each dataset.

### 5.2 Zero-Shot Forecasting (Out-of-Distribution)

MOIRAI was compared against state-of-the-art (SOTA) "full-shot" models (models trained specifically on the target data).

- **Probabilistic Forecasting:** On datasets like Electricity and Weather, MOIRAI-Base and Large achieved CRPS (Continuous Ranked Probability Score) results competitive with or better than PatchTST and TiDE, despite having never seen the target data during training.
- **Long-Sequence Forecasting:** MOIRAI consistently ranked among the top performers on ETT and Weather datasets, demonstrating its ability to capture long-term dependencies across different domains.

---

## 6. Key Insights and Ablations

The authors' ablation studies yielded several critical findings:

- **Any-variate Attention vs. Learned Embeddings:** Using binary attention bias outperformed traditional learned variate embeddings by a significant margin.
- **The Power of LOTSA:** Training only on smaller datasets (like GluonTS/Monash) resulted in a 23% performance drop compared to training on the full LOTSA archive.
- **Context Length Scalability:** Unlike many previous time series Transformers, MOIRAI's performance continues to improve as the context window increases, handling thousands of time steps effectively.

---

## 7. Model Variants

| Model Size | Parameters | Layers | Hidden Dim | Heads |
|------------|------------|--------|------------|-------|
| Small     | 14M       | 6     | 384       | 6    |
| Base      | 91M       | 12    | 768       | 12   |
| Large     | 311M      | 24    | 1024      | 16   |

---

## 8. Conclusion

MOIRAI marks a significant milestone in the development of Time Series Foundation Models. By solving the challenges of variable frequency and multivariate dimensionality through innovative patching and attention mechanisms, it provides a truly universal tool for forecasting that requires zero fine-tuning for most common applications.

**Future Directions:** The authors suggest that future iterations could move toward latent diffusion architectures or incorporate multi-modal inputs (e.g., text or tabular metadata) to further refine predictions.
