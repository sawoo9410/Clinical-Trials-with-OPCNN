# Clinical-Trials-with-OPCNN

Predicting clinical trial outcomes using **OPCNN** (Outer Product Convolutional Neural Network).

> Based on: [A Data-Driven Approach to Predicting Successes and Failures of Clinical Trials](https://www.frontiersin.org/articles/10.3389/fphar.2021.670670/full)
>
> Original PrOCTOR dataset: [kgayvert/PrOCTOR](https://github.com/kgayvert/PrOCTOR)

## Overview

This project implements OPCNN for binary classification of clinical trial outcomes (pass/fail). The model takes two input modalities — molecular descriptors (13 features) and body/tissue features (30 features) — and fuses them via an outer product operation followed by residual CNN blocks.

## Project Structure

```
├── src/
│   ├── main.py           # Entry point
│   ├── model.py          # OPCNN architecture
│   ├── Kfold_cv.py       # K-fold cross-validation (SMOTE + class weights)
│   ├── evaluation.py     # Evaluation metrics
│   ├── tf2_PrOCTOR_RandomForest.ipynb
│   └── tf2_PrOCTOR_SVM.ipynb
├── PrOCTOR_sample_data_all.csv   # Dataset (828 samples × 47 columns)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd src
python main.py
```

## Model Architecture

```
Des Input (13) ──→ Dense(50) ──→ Reshape(1,50) ──┐
                                                   ├─→ Outer Product ──→ Reshape(50,50,1)
Body Input (30) ──→ Dense(50) ──→ Reshape(1,50) ──┘
        │
        ↓
  3× Residual Conv2D Blocks (32 filters, 3×3)
        │
        ↓
  Flatten → Dense(50) → BN → Dense(50) → BN → Dense(1, sigmoid)
```

## Training

- **Cross-validation**: 10-fold
- **Class imbalance handling**: SMOTE oversampling + class weights
- **Optimizer**: Adam
- **Loss**: Binary crossentropy
- **Epochs**: 50

## Evaluation Metrics

ROC-AUC, AUPRC, Optimized Precision, Accuracy, Precision, Recall, F1, MCC, G-mean

## References

- Lo, A. et al. "Machine learning in chemoinformatics and drug discovery." *Drug discovery today* (2018).
- Gayvert, K. M. et al. "A data-driven approach to predicting successes and failures of clinical trials." *Cell chemical biology* (2016).
