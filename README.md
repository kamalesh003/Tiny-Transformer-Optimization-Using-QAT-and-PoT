# Tiny-Transformer-Optimization-Using-QAT-and-PoT


##  1. Overall Goal

The model is designed to create a **hardware-efficient Transformer** (a small ViT-like model) that:

* Uses **few parameters** (tiny architecture),
* Runs with **low precision** (4-bit or 6-bit quantization),
* Maintains **reasonable accuracy** despite compression,
* Can execute efficiently on **edge devices** using **streaming/FIFO-style inference**.

So, it’s a *hardware-aware neural network pipeline* — a bridge between machine learning and hardware deployment.

---

##  2. Model Structure and Flow

Conceptually, your model follows this structure:

```
Input image → Patch embedding → Transformer encoder blocks → Classifier → Output label
```

Let’s go step by step.

---

### (a) Input and Patch Embedding

* Each **CIFAR-10 image (32×32×3)** is split into small **patches** (say 4×4 or 8×8 pixels each).
* Each patch is **flattened and projected** into a vector (the embedding).

This projection gives a **sequence of embeddings**, one per patch — like “words” in a sentence for a Transformer.

**Goal:** Turn spatial image information into a sequence the Transformer can process.

**Assumption:** Local image features (edges, colors, textures) can be represented as patch tokens, similar to how words capture meaning in NLP.

---

### (b) Transformer Encoder

The Transformer encoder has **repeated layers (depth)**, each containing:

1. **Multi-Head Self-Attention (MHSA)**
   Each token (patch) “attends” to all other tokens — learns which other regions are important for context.

   * Each head learns different relationships (e.g., one head focuses on edges, another on texture).
   * Multiple heads = richer representation.

2. **Feedforward (MLP) layer**
   Each token’s feature vector is passed through a small two-layer network to refine its representation.

3. **Residual connections + Layer normalization**
   These stabilize learning and preserve information.

**Assumption:** The attention mechanism is powerful enough to capture spatial relationships in small images, even in a compressed form.

---

### (c) Classifier Head

After all the transformer blocks:

* The tokens are **averaged (or class-token extracted)** into one vector.
* That vector is passed through a **linear classifier** to predict the image’s label.

So conceptually:

> “The network learns to summarize all patches into one meaningful representation that maps to a class.”

---

## ⚙️ 3. Training Phases — Logic and Reasoning

The notebook you ran includes multiple *training and compression* phases, each motivated by specific trade-offs.

---

### **Phase 1: Neural Architecture Search (NAS)**

**Goal:** Find a small yet expressive configuration (embedding dim, depth, number of heads).

The logic:

* Larger models learn better but cost more.
* Smaller models are efficient but might underfit.
* So, a *search* (random or guided) finds the best balance.

Each configuration is trained briefly and evaluated — the best one is kept.

**Assumption:** Early validation accuracy predicts later full training performance.

---

### **Phase 2: Magnitude-Based Pruning**

**Goal:** Remove weights that have little contribution to output.

How it works conceptually:

* Each weight represents a “connection strength.”
* Small-magnitude weights (close to 0) contribute little.
* By zeroing them out, you remove redundancy.

This creates *sparsity* (some weights = 0) → less memory and faster computation.

**Assumption:** The network is over-parameterized — it can lose many weights without losing much accuracy.

---

### **Phase 3: Fine-Tuning After Pruning**

After pruning, the network’s representational capacity drops suddenly.
Fine-tuning helps **re-adapt surviving weights** to restore accuracy.

Conceptually:

> The network “relearns” to use its remaining connections effectively.

---

### **Phase 4: Quantization-Aware Training (QAT)**

**Goal:** Make the model use *low-precision arithmetic* (e.g., 4 bits) while staying accurate.

Conceptually:

* Each weight/activation value is rounded to the nearest representable number (e.g., powers of two for PoT quantization).
* This introduces *quantization noise* during training.
* The model learns to compensate for that noise.

Why important:

* 4-bit math runs 4–8× faster than 32-bit on specialized hardware.
* Memory footprint drops by the same factor.

**Assumption:** The model can learn to be robust against quantization noise if trained under those conditions.

---

### **Phase 5: FIFO-Streaming Inference Simulation**

**Goal:** Evaluate how the model would perform when tokens are processed sequentially — like in a streaming chip pipeline.

Processes tokens in small windows (tiles). Each tile is produced, consumed, then discarded. Only a few tiles are in memory at once. 
On real hardware (ASICs, FPGAs, or edge GPUs):

1. On-chip SRAM is limited — can’t store every token’s intermediate data.

2. External DRAM reads/writes are slow and power-hungry.

3. Throughput and latency depend on processing tokens as they arrive.

   
| Parameter         | Trade-off                                   |
| ----------------- | ------------------------------------------- |
| **tile_size ↑**   | Better accuracy, more memory                |
| **tile_size ↓**   | Lower memory, slightly reduced accuracy     |
| **fifo length ↑** | More latency hiding, more buffer memory     |
| **fifo length ↓** | Lower latency, but possible pipeline stalls |


Conceptually:

* Instead of processing the full token sequence at once, tokens are divided into **tiles** (small groups).
* Each tile flows through the network — only a few are held in memory.
* Results are passed forward as in a hardware FIFO (First-In, First-Out) buffer.

**Assumption:** Local temporal coherence between tokens allows partial processing without losing much global context.

**Purpose:** To mimic real-world inference hardware constraints (limited on-chip memory, need for low latency).

---

## 📊 4. Logical Flow (Conceptual Summary)

```
Image → Patches → Token embeddings
    ↓
[Transformer Block 1]
    ↓
[Transformer Block 2]
    ↓
... (multiple layers)
    ↓
Class token pooling → Linear classifier
    ↓
Predicted class (0–9)
```

Then:

1. **Train full model** for baseline accuracy.
2. **Prune** → make it sparse.
3. **Fine-tune** → recover accuracy.
4. **Quantize (4-bit PoT)** → compress further.
5. **FIFO simulate** → check streaming robustness.
6. **Evaluate final test accuracy.**

---

##  5. Underlying Assumptions and Theoretical Rationale

| Principle                   | Explanation                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Redundancy Hypothesis**   | Neural networks have many unnecessary parameters that can be pruned.                                    |
| **Quantization Robustness** | Neural weights and activations have limited dynamic range; low precision suffices with training.        |
| **Attention Efficiency**    | Transformer self-attention can capture global structure even in small token sequences.                  |
| **Locality of Tokens**      | Neighboring image patches are correlated — allows streaming and tiling without huge context loss.       |
| **Hardware Alignment**      | Model must be efficient not only in theory but in how it maps to real hardware memory and compute flow. |

---

# Inference on Test-set:

Using 'val_acc' for Validation Acc (Post-Fine-tune)

<img width="389" height="411" alt="image" src="https://github.com/user-attachments/assets/ebc034db-51fc-435c-a5a8-f0bf5ed125b2" />

# Visulization Result:
<img width="1189" height="690" alt="image" src="https://github.com/user-attachments/assets/7573d1cf-e9a5-4ca0-a0d9-a5067f9e53b4" />

# About Model:

```bash
Parameter Count: 413706
Model Size (bytes): 1654824
Measured Sparsity: 0.000000
```

