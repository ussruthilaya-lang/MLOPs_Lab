# Lab 3 – Data Lab - Fine Tuning Data Pipeline

# What I Learned From This Lab

## 1. Fine-Tuning Is More Than Just Training

Before this lab, I thought fine-tuning was mainly about choosing a model and training it.

Now I understand the full pipeline:

* Inspect data distribution
* Handle class imbalance
* Tokenize correctly
* Design the loss function
* Choose meaningful evaluation metrics

Every step changes what the model actually learns. High accuracy alone does not mean the model is correct.

---

## 2. Streaming Changes How You Think About Data

In earlier labs, stratified split was easy. But in a streaming-style setup, random indexing is not available.

So I:

* Shuffled with fixed seed
* Split sequentially (80/20)

This helped me understand that in real pipelines, you don’t always control perfect data splits. Shuffle buffer size and data order directly affect gradient updates.

---

## 3. Imbalance Affects Gradient Strength

The dataset was imbalanced:

* Neutral: 61.4%
* Positive: 25.2%
* Negative: 13.4%

If trained normally, the model would prefer predicting Neutral.

So I:

* Did a first pass to count labels
* Computed class weights
* Used weighted `CrossEntropyLoss`

This increased recall for the minority class (Negative).

I understood that imbalance is not just a dataset issue — it is a gradient magnitude issue.

---

## 4. Distributed Training Requires Structural Awareness

Even though I ran on single CPU, I implemented DDP-ready code:

* `rank` and `world_size` variables
* Learning rate scaling
* Rank-0 checkpoint saving
* Rank logging

Example log:

```id="ddp1"
[train.py] Rank 0 processing 1811 samples
```

This made me understand that in multi-GPU systems:

* Each process sees different data shards
* Gradients are synchronized
* Only one process should save checkpoints

Distributed training is not automatic — it requires correct structure.

---

## 5. Production Structure Matters

I moved from notebook to modular scripts:

```id="proj1"
finrisk/
├── data.py
├── model.py
├── train.py
├── evaluate.py
├── utils.py
├── config.py
└── app.py
```

Each file has one responsibility.
Hyperparameters are centralized in `config.py`.
The model can be retrained, evaluated, or served independently.

This felt much closer to real MLOps practice.

---

# Customizations I Made

## 1. Independent Dataset and Use Case

Instead of reusing lab datasets, I chose Financial PhraseBank and framed it as a financial risk classifier.

This made the project more realistic and domain-focused.

---

## 2. Two-Pass Weight Computation

Instead of assuming known class distribution, I:

* First pass → count labels
* Second pass → train with weights

This reflects streaming-style constraints.

---

## 3. Weighted Loss for Minority Recall

Used weighted `CrossEntropyLoss`:

* Negative: 2.49×
* Positive: 1.32×
* Neutral: 0.54×

Result:

Macro F1 ≈ 0.94
Negative recall ≈ 0.98

This shows imbalance correction worked.

---

## 4. Reproducibility and Clean Engineering

Added:

* `set_seed(42)` across random, numpy, torch
* Centralized hyperparameters
* Rank-0 checkpoint saving
* Streamlit UI with confidence scores

The model is not just trained — it is deployable.

---

# How To Run (For TA)

### Step 1 – Setup Environment

```bash
python -m venv finrisk-env
finrisk-env\Scripts\activate   # Windows
source finrisk-env/bin/activate  # Mac/Linux
pip install torch transformers datasets scikit-learn streamlit
```

---

### Step 2 – Train

```bash
cd finrisk
python train.py
```

Expected logs:

```id="trainlog"
[data.py] Label distribution estimated. Class weights: [2.49 0.54 1.32]
[train.py] Rank 0 of 1 — device: cpu
Epoch 1/3 — Loss: ~0.63
Epoch 2/3 — Loss: ~0.17
Epoch 3/3 — Loss: ~0.07
[train.py] Checkpoint saved to checkpoint.pt
```

---

### Step 3 – Evaluate

```bash
python evaluate.py
```

Expected macro F1 ≈ 0.94.

---

### Step 4 – Run App

```bash
streamlit run app.py
```

The UI allows text input and shows predicted class with confidence scores.

---

# Final Reflection

This lab helped me see that ML engineering is mostly about decisions outside the model:

* Shuffle strategy
* Loss weighting
* Reproducibility
* Proper structure
* Deployment readiness

One key insight was label interpretation.
When testing “filed for bankruptcy,” the model predicted Neutral with high confidence. That seemed wrong at first, but the dataset labels reflect financial reporting tone, not emotional sentiment. The model learned the dataset definition correctly.

Overall, this lab connected model training, distributed awareness, and deployment into one coherent system.
