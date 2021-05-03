from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, confusion_matrix, accuracy_score, matthews_corrcoef, accuracy_score
from imblearn.metrics import geometric_mean_score
import math
import numpy as np

def pred_reshape(predict):
    y_predict = []
    for i in range(len(predict)):
        y_predict = np.append(y_predict, predict[i])
    
    y_predict = np.array(y_predict)
    y_predict = y_predict.reshape(y_predict.shape[0],-1)
    return y_predict

class Evaluation:
    def __init__(self, pred, y):
        self.y_pred = pred_reshape(pred)
        self.y = y
        
    def matrix(self):
        y_1 = self.y.reshape(self.y.shape[0],-1)
        y_pred = np.array(self.y_pred)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        
        y_classify = []
        for i in range(len(self.y_pred)):
            if self.y_pred[i] >= 0.5:
                a = 1.
                y_classify.append(a)
            else:
                a = 0.
                y_classify.append(a)
        
        fpr,tpr,threshold = roc_curve(y_1 , y_pred, pos_label = 1)
        precision, recall, threshold = precision_recall_curve(y_1, y_pred, pos_label = 1)
        
        roc_auc = auc(fpr,tpr)
        auprc = auc(recall, precision)
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)
        F1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
        # binary 
        accuracy = accuracy_score(y_1, y_classify)
        mcc = matthews_corrcoef(y_1, y_classify) 
        g_mean = geometric_mean_score(y_1, y_classify)
        confusion = confusion_matrix(y_1, y_classify)
        
        tn, fp, fn, tp = confusion.ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        ppv = tp / (tp + fp)
        fnr = fn / (fn + tp)
        fpr = fp / (fp + tn)

        confu_precision = ppv 
        confu_recall = tpr # sensitivity
        confu_f1 = 2 * ((ppv * tpr) / (ppv + tpr))
        confu_accuracy = (tp + tn) / (tp + tn + fp + fn)
        confu_mcc = ((tp * tn)-(fp-fn))/ math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
        confu_g_mean = math.sqrt(tpr * tnr)
        Optimized_precision = (confu_accuracy - abs(tnr-tpr)) / (tnr + tpr)

        print('공통 \nAUC :',roc_auc) # pb
        print("AUPRC :", auprc) # pb
        print("Optimized precision :", Optimized_precision)

        print("\nfunction 사용\nAccuracy :", accuracy)
        print("Precision(pb) :", mean_precision) #pb
        print("Recall(pb) :", mean_recall) # pb
        print("F1 score(pb) :", F1) #pb
        print("MCC :", mcc)
        print("G-mean :", g_mean)
        
        print("\nConfusion_matrix 사용 \n", confusion)
        print("Accuracy :", confu_accuracy)
        print("Precision :", confu_precision) 
        print("Recall :", confu_recall) 
        print("F1 score :", confu_f1) 
        print("MCC :", confu_mcc)
        print("G-mean :", confu_g_mean)
        print("TPR :", tpr)
        print("TNR :", tnr)
        
        return roc_auc, auprc, Optimized_precision, accuracy, mean_precision, mean_recall, F1, mcc, g_mean, confu_accuracy, confu_precision, confu_recall, confu_f1, confu_mcc, confu_g_mean;