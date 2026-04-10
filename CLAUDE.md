# CLAUDE.md

## Project Overview
Clinical trial outcome prediction using OPCNN (Outer Product CNN).
Based on PrOCTOR dataset (828 samples, 47 features).

## Structure
- `src/main.py` — Entry point: loads data, trains OPCNN, evaluates
- `src/model.py` — OPCNN model architecture (dual-input CNN with outer product fusion)
- `src/Kfold_cv.py` — K-fold cross-validation with SMOTE + class weighting
- `src/evaluation.py` — Metrics: ROC-AUC, AUPRC, MCC, G-mean, etc.
- `PrOCTOR_sample_data_all.csv` — Dataset

## Run
```bash
cd src
python main.py
```

## Dependencies
```bash
pip install -r requirements.txt
```

## Key Details
- Dataset split: 13 descriptor features + 30 body features
- Model: Dual-input → Dense → Outer Product → 3x ResNet Conv2D blocks → Dense → Sigmoid
- Training: 10-fold CV, SMOTE oversampling, class weights, 50 epochs
- Output: Binary classification (passed/failed clinical trial)

## Conventions
- Git commits: Do NOT include Co-Authored-By lines
- Language: Code in English, comments in English
