# DA6401 - Assignment 3

Wandb Report: https://wandb.ai/ce21b031/DA6401%20-%20Assignment3/reports/DA-6401-Assignment-3--VmlldzoxMjg1NDAyMg?accessToken=dh1dla1bnyldxwflsbga5k5w3ogewfdfoyvf3q70pmguiubrtyyqtbyxp60zqwa4

This repository provides a comprehensive pipeline for Hindi transliteration using sequence-to-sequence (Seq2Seq) neural models, including both vanilla and attention-based architectures. It is designed for research, experimentation, and visualization on the [Dakshina dataset](https://github.com/google-research-datasets/dakshina), and includes tools for training, evaluation, hyperparameter sweeps, and detailed attention visualization.

---

## **Features**

- **Data Preparation:** Robust loading, cleaning, and batching for the Dakshina dataset.
- **Seq2Seq Models:** Implements both vanilla and attention-based Seq2Seq models with configurable RNN, GRU, or LSTM cells.
- **Training & Evaluation:** Modular scripts for training, validation, and test evaluation, with token-level accuracy reporting.
- **Hyperparameter Sweeps:** Integration with [Weights & Biases (wandb)](https://wandb.ai/) for automated hyperparameter optimization.
- **Visualization:** 
  - Generates HTML and PNG visualizations of model predictions and attention maps.
  - Supports Hindi-compatible fonts for accurate Devanagari rendering.
- **Easy Extensibility:** Modular codebase for adapting to other language pairs or datasets.

---

## **Directory Structure**

```
.
├── connectivity.py
├── data_prep.py
├── decoder.py
├── encoder.py
├── eval.py
├── save_attention_test.py
├── save_test.py
├── seq2seq.py
├── train_attention_wandb.py
├── train_wandb.py
├── dakshina_dataset_v1.0/hi/lexicons/
│   ├── hi.translit.sampled.train.tsv
│   ├── hi.translit.sampled.dev.tsv
│   └── hi.translit.sampled.test.tsv
└── predictions_attention/
    ├── predictions_attention.csv
    └── attention_grid.png
```

---

## **Getting Started**

### **1. Requirements**

- Python 3.7+
- PyTorch
- pandas
- matplotlib
- wandb
- numpy

Install dependencies using pip:

```bash
pip install torch pandas matplotlib wandb numpy
```

### **2. Dataset**

Download the [Dakshina dataset](https://github.com/google-research-datasets/dakshina) and place the Hindi lexicon files under `dakshina_dataset_v1.0/hi/lexicons/`.

### **3. Training**

#### **Vanilla Seq2Seq Model**

To train a vanilla Seq2Seq model (without attention) and run a hyperparameter sweep:

```bash
python train_wandb.py
```

- Adjust sweep configuration as needed in `train_wandb.py`.
- The best model is saved as `best_model.pth`.

#### **Seq2Seq with Attention**

To train an attention-based Seq2Seq model and run a sweep:

```bash
python train_attention_wandb.py
```

- The best attention model is saved as `best_attention_model.pth`.

---

### **4. Evaluation & Visualization**

#### **Vanilla Model Prediction and Visualization**

```bash
python save_test.py
```

- Saves predictions to `predictions_vanilla/` as `.txt` and `.csv`.
- Generates a 3x3 grid of random predictions (`prediction_table.png`).

#### **Attention Model Prediction and Visualization**

```bash
python save_attention_test.py
```

- Saves predictions and attention weights to `predictions_attention/`.
- Generates an attention heatmap grid (`attention_grid.png`).
- Saves all predictions as a CSV.

#### **HTML Attention Visualization**

```bash
python connectivity.py
```

- Generates HTML files for the first 20 test samples in `attention_html/` showing character-level attention alignment.

---

## **Code Overview**

| File                        | Purpose                                                                                      |
|-----------------------------|----------------------------------------------------------------------------------------------|
| `data_prep.py`              | Data loading, cleaning, vocabulary building, and batching.                                   |
| `encoder.py` / `decoder.py` | Encoder and decoder modules for Seq2Seq models (with and without attention).                 |
| `seq2seq.py`                | Seq2Seq model wrappers for vanilla and attention-based architectures.                        |
| `train_wandb.py`            | Training and hyperparameter sweep for vanilla Seq2Seq.                                       |
| `train_attention_wandb.py`  | Training and sweep for attention-based Seq2Seq.                                              |
| `eval.py`                   | Evaluation utilities for both model types.                                                   |
| `save_test.py`              | Prediction and visualization for vanilla Seq2Seq.                                            |
| `save_attention_test.py`    | Prediction, attention visualization, and CSV export for attention model.                     |
| `connectivity.py`           | HTML visualization of attention for individual samples.                                      |

---

## **Model Configuration**

- **Embedding Size:** 64, 128, 256 (sweepable)
- **Hidden Size:** 128, 256, 512 (sweepable)
- **Num Layers:** 1, 2, 3 (sweepable)
- **RNN Cell:** RNN, GRU, LSTM (sweepable)
- **Dropout:** 0.2, 0.3 (sweepable)
- **Batch Size:** 64
- **Learning Rate:** 1e-4 to 1e-3 (sweepable)
- **Teacher Forcing Ratio:** 0.5 (default)

---

## **Attention Visualization**

- **PNG Grids:** Random sample predictions with attention heatmaps (`attention_grid.png`).
- **HTML Files:** Per-sample attention alignment, color-coded by attention strength.
- **Font Support:** Attempts to use a Devanagari-compatible font for correct Hindi rendering. If unavailable, warns user.

---

## **Weights & Biases Integration**

- All training scripts log metrics and hyperparameters to wandb.
- Sweeps can be launched directly from the training scripts.
- Best models are automatically checkpointed.

---

## **Extending to Other Languages**

- Change the `data_dir` and file paths in the scripts to point to the desired language's lexicon files from the Dakshina dataset.
- The vocabulary and dataset classes are language-agnostic.

---