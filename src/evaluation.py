"""
evaluation.py - Comprehensive evaluation metrics for binary classification.

Computes two sets of metrics:
  1. Probability-based: ROC-AUC, AUPRC, Optimized Precision
     (computed from raw prediction probabilities)
  2. Threshold-based: Accuracy, Precision, Recall, F1, MCC, G-mean
     (computed after applying 0.5 threshold to convert probabilities to classes)

Both sklearn-based and confusion-matrix-based calculations are provided
for cross-validation of metric correctness.
"""

from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, accuracy_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
import math
import numpy as np


def pred_reshape(predict):
    """
    Reshape K-fold prediction dict into a contiguous numpy array.

    K-fold CV returns predictions as {index: value} dict. This function
    converts it into an ordered array sorted by sample index.

    Args:
        predict: dict of {sample_index: prediction_value} from K_fold methods.

    Returns:
        np.ndarray of shape (n_samples, 1) with predictions in original order.
    """
    y_predict = []
    for i in range(len(predict)):
        y_predict = np.append(y_predict, predict[i])

    y_predict = np.array(y_predict)
    y_predict = y_predict.reshape(y_predict.shape[0],-1)
    return y_predict


class Evaluation:
    """
    Evaluation metrics calculator for binary classification results.

    Args:
        pred: Prediction dict from K_fold (converted via pred_reshape).
        y: True labels, shape (n_samples, 1). Binary: 0 or 1.
    """
    def __init__(self, pred, y):
        self.y_pred = pred_reshape(pred)
        self.y = y

    def matrix(self):
        """
        Compute and print all evaluation metrics.

        Metrics computed:
          - ROC-AUC: Area under ROC curve (probability-based)
          - AUPRC: Area under Precision-Recall curve (probability-based)
          - Optimized Precision: Accuracy adjusted for class balance
          - Accuracy, Precision, Recall, F1: Standard classification metrics
          - MCC: Matthews Correlation Coefficient (balanced metric for imbalanced data)
          - G-mean: Geometric mean of sensitivity and specificity

        Returns:
            tuple: (roc_auc, auprc, optimized_precision,
                    accuracy, mean_precision, mean_recall, F1, mcc, g_mean,
                    confu_accuracy, confu_precision, confu_recall, confu_f1,
                    confu_mcc, confu_g_mean)
        """
        y_1 = self.y.reshape(self.y.shape[0],-1)
        y_pred = np.array(self.y_pred)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)

        # Convert probabilities to binary predictions (threshold = 0.5)
        y_classify = []
        for i in range(len(self.y_pred)):
            if self.y_pred[i] >= 0.5:
                a = 1.
                y_classify.append(a)
            else:
                a = 0.
                y_classify.append(a)

        # Probability-based curves
        fpr,tpr,threshold = roc_curve(y_1 , y_pred, pos_label = 1)
        precision, recall, threshold = precision_recall_curve(y_1, y_pred, pos_label = 1)

        # Probability-based metrics
        roc_auc = auc(fpr,tpr)
        auprc = auc(recall, precision)
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)
        F1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)

        # Threshold-based metrics (using sklearn functions)
        accuracy = accuracy_score(y_1, y_classify)
        mcc = matthews_corrcoef(y_1, y_classify)
        g_mean = geometric_mean_score(y_1, y_classify)
        confusion = confusion_matrix(y_1, y_classify)

        # Confusion matrix decomposition
        tn, fp, fn, tp = confusion.ravel()
        tpr = tp / (tp + fn)          # True Positive Rate (Sensitivity)
        tnr = tn / (tn + fp)          # True Negative Rate (Specificity)
        ppv = tp / (tp + fp)          # Positive Predictive Value (Precision)
        fnr = fn / (fn + tp)          # False Negative Rate
        fpr = fp / (fp + tn)          # False Positive Rate

        # Confusion-matrix-based metrics (manual calculation for verification)
        confu_precision = ppv
        confu_recall = tpr  # Same as sensitivity
        confu_f1 = 2 * ((ppv * tpr) / (ppv + tpr))
        confu_accuracy = (tp + tn) / (tp + tn + fp + fn)
        confu_mcc = ((tp * tn)-(fp * fn))/ math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
        confu_g_mean = math.sqrt(tpr * tnr)
        # Optimized Precision: penalizes accuracy by the gap between TPR and TNR
        Optimized_precision = (confu_accuracy - abs(tnr-tpr)) / (tnr + tpr)

        print('--- Probability-based ---')
        print('AUC :', roc_auc)
        print("AUPRC :", auprc)
        print("Optimized precision :", Optimized_precision)

        print("\n--- Sklearn metrics ---")
        print("Accuracy :", accuracy)
        print("Precision :", mean_precision)
        print("Recall :", mean_recall)
        print("F1 score :", F1)
        print("MCC :", mcc)
        print("G-mean :", g_mean)

        print("\n--- Confusion matrix ---\n", confusion)
        print("Accuracy :", confu_accuracy)
        print("Precision :", confu_precision)
        print("Recall :", confu_recall)
        print("F1 score :", confu_f1)
        print("MCC :", confu_mcc)
        print("G-mean :", confu_g_mean)
        print("TPR :", tpr)
        print("TNR :", tnr)

        return roc_auc, auprc, Optimized_precision, accuracy, mean_precision, mean_recall, F1, mcc, g_mean, confu_accuracy, confu_precision, confu_recall, confu_f1, confu_mcc, confu_g_mean;
