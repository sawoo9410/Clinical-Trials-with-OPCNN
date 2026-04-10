"""
main.py - Entry point for OPCNN clinical trial outcome prediction.

Pipeline:
  1. Load PrOCTOR dataset (828 clinical trials × 47 features)
  2. Preprocess: fill NaN with median, encode target, normalize
  3. Split features into descriptor (13) and body (30) modalities
  4. Build OPCNN model and run 10-fold CV with SMOTE + class weights
  5. Evaluate and print comprehensive metrics
"""

import numpy as np
import pandas as pd
from model import network
from evaluation import Evaluation
import Kfold_cv

from sklearn.preprocessing import StandardScaler


# --- 1. Data Loading ---
data = pd.read_csv('PrOCTOR_sample_data_all.csv', header=0)

# --- 2. Preprocessing ---
# Fill missing values with column median (only for numeric feature columns)
data1 = data.fillna(data.median()['MolecularWeight':'Salivary Gland'])
# Encode target: "passed" -> 1, all others -> 0
data1["target"] = np.where(data1.iloc[:, 1] == "passed", 1, 0)

# Standardize features (zero mean, unit variance)
scaler = StandardScaler()
X = data1.iloc[:, 2:-1]  # All feature columns (excluding ID, Label, target)
y = data1['target']

X = np.array(X)
X = scaler.fit_transform(X)
y = np.array(y)
reshape_y = y.reshape(y.shape[0],-1)  # (n,) -> (n, 1) for evaluation

# --- 3. Modality Split ---
des=X[:,:13]       # Descriptor features: MolecularWeight, LogP, etc. (13 cols)
body=X[:,13:]      # Body/tissue expression features (30 cols)
_, des_col = des.shape
_, body_col = body.shape

# --- 4. Model Building & Training ---
hid_layer = [100, 100, 50]  # Hidden layer sizes for the network

OPCNN = network(des_col, body_col, hid_layer, 1).OPCNN()
OPCNN = Kfold_cv.K_fold(X, y, OPCNN, epochs= 50, kfold=10)

# Run 10-fold CV with SMOTE oversampling + class weights (baseline: .base())
pred = OPCNN.base()

# --- 5. Evaluation ---
OPCNN_result = Evaluation(pred, reshape_y).matrix()
