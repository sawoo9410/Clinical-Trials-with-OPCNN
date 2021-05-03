import numpy as np
import pandas as pd
from model import network
from evaluation import Evaluation
import Kfold_cv

from sklearn.preprocessing import StandardScaler


data = pd.read_csv('PrOCTOR_sample_data_all.csv', header=0)
data1 = data.fillna(data.median()['MolecularWeight':'Salivary Gland'])
data1["target"] = np.where(data1.iloc[:, 1] == "passed", 1, 0)

scaler = StandardScaler()
X = data1.iloc[:, 2:-1]
y = data1['target']

X = np.array(X)
X = scaler.fit_transform(X)
y = np.array(y)
reshape_y = y.reshape(y.shape[0],-1)

des=X[:,:13]
body=X[:,13:]
_, des_col = des.shape
_, body_col = body.shape

hid_layer = [100, 100, 50]

OPCNN = network(des_col, body_col, hid_layer, 1).OPCNN()
OPCNN = Kfold_cv.K_fold(X, y, OPCNN, epochs= 50, kfold=10)

pred = OPCNN.base()

OPCNN_result = Evaluation(pred, reshape_y).matrix()