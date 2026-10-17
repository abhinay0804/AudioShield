# Architecture Reference

## Multi-Granularity Temporal Attention for Explainable Audio Deepfake Detection

---

## High-Level Architecture Diagram

```
                           RAW AUDIO (.flac)
                          ASVspoof 2019 LA
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING                                │
│  • Resample to 16 kHz                                                │
│  • Trim leading/trailing silence (energy-based VAD)                  │
│  • Normalize amplitude to [-1, 1]                                    │
│  • Pad/truncate to 4 seconds (64,000 samples)                       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      LFCC FEATURE EXTRACTION                         │
│  • Linear Frequency Cepstral Coefficients                            │
│  • 20 static + 20 Δ + 20 ΔΔ = 60 dimensions                        │
│  • N_FFT=512, window=20ms (320 samples), hop=10ms (160 samples)     │
│  • Output shape: [B, T≈400, 60]                                     │
│                                                                      │
│  + SpecAugment (training only):                                      │
│    • Time masking: mask 10–20% of random frames                      │
│    • Frequency masking: mask 2–4 random coefficient bands            │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    LIGHTWEIGHT CNN FRONT-END                          │
│                                                                      │
│  Reshape: [B, T, 60] → [B, 1, T, 60]  (treat as 1-channel image)   │
│                                                                      │
│  Block 1: Conv2D(1→32, 3×3, pad=1) → BN2D(32) → LeakyReLU(0.2)    │
│           → MaxPool2D(2×2)                                           │
│           Output: [B, 32, T/2, 30]                                   │
│                                                                      │
│  Block 2: Conv2D(32→64, 3×3, pad=1) → BN2D(64) → LeakyReLU(0.2)   │
│           → MaxPool2D(2×2)                                           │
│           Output: [B, 64, T/4, 15]                                   │
│                                                                      │
│  Reshape: [B, 64, T/4, 15] → [B, T/4, 64×15] = [B, T'≈100, 960]   │
│  Project: Linear(960, 128)                                           │
│  Output:  [B, T'≈100, 128]                                          │
│                                                                      │
│  Purpose: Extract local spectro-temporal patterns, reduce dims       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
┌─────────────────────────────┐  ┌─────────────────────────────────────┐
│   🔬 FINE-GRAINED STREAM    │  │   🔭 COARSE-GRAINED STREAM          │
│   (Frame-level, ~40ms       │  │   (Segment-level, ~500ms)           │
│    after CNN downsampling)  │  │                                     │
│                             │  │  Temporal Avg Pool:                  │
│  Input: [B, T'≈100, 128]   │  │    window=12, stride=12             │
│                             │  │    [B, T'≈100, 128]                 │
│  BiLSTM Layer 1:            │  │    → [B, T''≈8, 128]               │
│    nn.LSTM(                 │  │                                     │
│      input_size=128,        │  │  BiLSTM Layer:                      │
│      hidden_size=128,       │  │    nn.LSTM(                         │
│      bidirectional=True,    │  │      input_size=128,                │
│      batch_first=True       │  │      hidden_size=64,                │
│    )                        │  │      bidirectional=True,             │
│    → [B, T', 256]           │  │      batch_first=True               │
│    + LayerNorm(256)         │  │    )                                │
│    + Dropout(0.25)          │  │    → [B, T'', 128]                  │
│                             │  │    + LayerNorm(128)                  │
│  BiLSTM Layer 2:            │  │    + Dropout(0.25)                   │
│    nn.LSTM(                 │  │                                     │
│      input_size=256,        │  │  Multi-Head Self-Attention:          │
│      hidden_size=64,        │  │    n_heads=2, d_model=128, d_k=32  │
│      bidirectional=True,    │  │    Q = K = V = BiLSTM output        │
│      batch_first=True       │  │    Attn(Q,K,V) = softmax(QKᵀ/√d_k)V│
│    )                        │  │    → attn_weights: [B, 2, 8]        │
│    → [B, T', 128]           │  │    → Weighted sum across T''        │
│    + LayerNorm(128)         │  │    → coarse_context: [B, 128]       │
│    + Dropout(0.25)          │  │                                     │
│                             │  │  Captures:                          │
│  Multi-Head Self-Attention: │  │    • Prosody patterns               │
│    n_heads=4, d_model=128,  │  │    • Speaking rate anomalies        │
│    d_k=32                   │  │    • Macro-level rhythm artifacts   │
│    Q = K = V = BiLSTM output│  │                                     │
│    → attn_weights: [B,4,100]│  │                                     │
│    → Weighted sum across T' │  │                                     │
│    → fine_context: [B, 128] │  │                                     │
│                             │  │                                     │
│  Captures:                  │  │                                     │
│    • Micro-glitches         │  │                                     │
│    • Phoneme transitions    │  │                                     │
│    • Frame-level artifacts  │  │                                     │
└──────────────┬──────────────┘  └────────────────┬────────────────────┘
               │                                  │
               └──────────────┬───────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FUSION + CLASSIFICATION                            │
│                                                                      │
│  Concatenate: [fine_context ∥ coarse_context] → [B, 256]            │
│                                                                      │
│  FC1: Linear(256, 128) → LayerNorm(128) → ReLU → Dropout(0.4)      │
│  FC2: Linear(128, 2)                                                 │
│                                                                      │
│  AM-Softmax applied on FC2 output:                                   │
│    • Normalize weights W and features x                              │
│    • cosθ = W̃ᵀx̃                                                     │
│    • Subtract margin m=0.35 from correct class cosine                │
│    • Scale by s=30                                                   │
│    • Output: scaled logits for cross-entropy                         │
│                                                                      │
│  Embedding output (256-d fusion vector) saved for:                   │
│    • SupCon loss computation during training                         │
│    • t-SNE / UMAP visualization during evaluation                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Shape Flow Summary

| Stage | Shape | Notes |
|-------|-------|-------|
| Raw audio | `[B, 64000]` | 4s × 16kHz |
| LFCC extraction | `[B, ~400, 60]` | T frames × 60 coefficients |
| CNN input (reshape) | `[B, 1, ~400, 60]` | 1-channel "image" |
| After CNN block 1 | `[B, 32, ~200, 30]` | MaxPool halves both dims |
| After CNN block 2 | `[B, 64, ~100, 15]` | MaxPool halves again |
| CNN output (reshape + project) | `[B, ~100, 128]` | Flatten freq dim → project to 128 |
| Fine stream BiLSTM-1 | `[B, ~100, 256]` | 128×2 (bidirectional) |
| Fine stream BiLSTM-2 | `[B, ~100, 128]` | 64×2 (bidirectional) |
| Fine attention weights | `[B, 4, ~100]` | 4 heads × T' time steps |
| Fine context vector | `[B, 128]` | Attention-weighted sum |
| Coarse pool | `[B, ~8, 128]` | 100/12 ≈ 8 segments |
| Coarse BiLSTM | `[B, ~8, 128]` | 64×2 (bidirectional) |
| Coarse attention weights | `[B, 2, ~8]` | 2 heads × 8 segments |
| Coarse context vector | `[B, 128]` | Attention-weighted sum |
| Fusion | `[B, 256]` | Concatenate fine + coarse |
| Classifier output | `[B, 2]` | 2-class logits |

---

## Loss Functions

### 1. AM-Softmax (Primary)

$$\mathcal{L}_{\text{AM}} = -\log \frac{e^{s(\cos\theta_{y_i} - m)}}{e^{s(\cos\theta_{y_i} - m)} + \sum_{j \neq y_i} e^{s \cos\theta_j}}$$

Where:
- $s = 30$ (scale factor)
- $m = 0.35$ (angular margin)
- $\theta_{y_i}$ = angle between feature vector and correct class weight vector
- Normalizes both weights and features before computing cosine

### 2. SupCon Loss (Auxiliary)

$$\mathcal{L}_{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

Where:
- $\mathbf{z}_i$ = L2-normalized 256-d fusion embedding
- $P(i)$ = set of samples with same label as $i$ in the batch
- $\tau = 0.07$ (temperature)

### 3. Combined Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{AM}} + 0.1 \cdot \mathcal{L}_{\text{SupCon}}$$

---

## Multi-Head Self-Attention (Custom Implementation)

```
Input: H ∈ ℝ^{B × T × d_model}   (d_model = 128)

For each head h ∈ {1, ..., n_heads}:
    Q_h = H @ W_Q^h    ∈ ℝ^{B × T × d_k}     (d_k = 32)
    K_h = H @ W_K^h    ∈ ℝ^{B × T × d_k}
    V_h = H @ W_V^h    ∈ ℝ^{B × T × d_v}     (d_v = 32)

    scores_h = (Q_h @ K_h^T) / √d_k    ∈ ℝ^{B × T × T}
    α_h = softmax(scores_h, dim=-1)     ∈ ℝ^{B × T × T}

    ★ STORE α_h for explainability ★

    context_h = α_h @ V_h               ∈ ℝ^{B × T × d_v}

Concatenate all heads:
    multi_head = [context_1 ∥ ... ∥ context_n] ∈ ℝ^{B × T × (n_heads × d_v)}

Project:
    output = multi_head @ W_O            ∈ ℝ^{B × T × d_model}

Temporal pooling (attention-weighted sum across T):
    importance = mean(α_h, dim=heads)    ∈ ℝ^{B × T}  (averaged across heads)
    context_vector = sum(importance * output, dim=T)   ∈ ℝ^{B × d_model}
```

---

## Hyperparameters Quick Reference

| Category | Parameter | Value |
|----------|-----------|-------|
| **Audio** | Sample rate | 16,000 Hz |
| | Max length | 4.0 seconds |
| | N_FFT | 512 |
| | Window | 20ms (320 samples) |
| | Hop | 10ms (160 samples) |
| **LFCC** | Static coefficients | 20 |
| | With Δ + ΔΔ | 60 total |
| **CNN** | Filters | [32, 64] |
| | Kernel | 3×3, padding=1 |
| | Pooling | MaxPool 2×2 |
| | Activation | LeakyReLU(0.2) |
| **Fine BiLSTM** | Layers | 2 |
| | Hidden sizes | [128, 64] per direction |
| | Output dims | [256, 128] |
| **Coarse BiLSTM** | Layers | 1 |
| | Hidden size | 64 per direction |
| | Pool window | 12 (→ ~8 segments) |
| **Fine Attention** | Heads | 4 |
| | d_k, d_v | 32 |
| **Coarse Attention** | Heads | 2 |
| | d_k, d_v | 32 |
| **Classifier** | FC sizes | [256→128→2] |
| | Dropout | 0.4 |
| **AM-Softmax** | Margin (m) | 0.35 |
| | Scale (s) | 30 |
| **SupCon** | Temperature (τ) | 0.07 |
| | Weight (λ) | 0.1 |
| **Training** | Optimizer | AdamW |
| | Learning rate | 1e-3 |
| | Weight decay | 1e-4 |
| | Scheduler | CosineAnnealing (T_max=30, η_min=1e-6) |
| | Batch size | 32 |
| | Max epochs | 35 |
| | Early stopping | patience=6 on dev EER |
| | Gradient clip | max_norm=1.0 |
| | Mixed precision | Yes (torch.cuda.amp) |
| **SpecAugment** | Time mask | 10–20% of frames |
| | Freq mask | 2–4 bands |
