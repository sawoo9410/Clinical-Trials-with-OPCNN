from sklearn.model_selection import KFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import keras
import numpy as np

sm = SMOTE(random_state=202004)

class K_fold:
    def __init__(self, X, y, model, epochs, kfold, tfn=None):
        self.X = X
        self.y = y
        self.model = model
        self.epochs = epochs
        self.kfold = KFold(n_splits = kfold, shuffle = True, random_state = 202004)
        self.des = X[:,:13]
        self.body = X[:,13:]
        self.TFL = tfn
        
    ''' None '''    
    def base(self):
        base_y = dict()
        
        for train_index, test_index in self.kfold.split(self.X):
            print("TEST:", test_index)  
            des_train, des_test = self.des[train_index], self.des[test_index]
            body_train, body_test = self.body[train_index], self.body[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.model.compile(optimizer= keras.optimizers.Adam(), loss='binary_crossentropy', metrics = ['accuracy'])
            
            if self.TFL is None:
                kf_history = self.model.fit(x = [des_train, body_train], y = y_train, epochs=self.epochs)
                y_pred = self.model.predict([des_test[:], body_test[:]])
            else:
                kf_history = self.model.fit(x = [np.ones(des_train.shape[0]),des_train, body_train], y = y_train, epochs=self.epochs)
                y_pred = self.model.predict([np.ones(des_test.shape[0]), des_test[:], body_test[:]])
            
            for i in range(len(test_index)):
                index = test_index[i]
                base_y[index] = y_pred[i]
            
        return base_y
    
    ''' Class Weight '''
    def weight(self):
        cw_y = dict()
        
        for train_index, test_index in self.kfold.split(self.X):
            print("TEST:", test_index)
            des_train, des_test = self.des[train_index], self.des[test_index]
            body_train, body_test = self.body[train_index], self.body[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            neg, pos = np.bincount(y_train)
            total = neg + pos

            weight_for_0 = (1 / neg)*(total)/2.0 
            weight_for_1 = (1 / pos)*(total)/2.0

            class_weight = {0: weight_for_0, 1: weight_for_1}
            
            self.model.compile(optimizer= keras.optimizers.Adam(), loss='binary_crossentropy', metrics = ['accuracy'])
            
            if self.TFL is None:
                kf_history = self.model.fit(x = [des_train, body_train], y = y_train, epochs=self.epochs, class_weight = class_weight)
                y_pred = self.model.predict([des_test[:], body_test[:]])
            else:
                kf_history = self.model.fit(x = [np.ones(des_train.shape[0]), des_train, body_train], y = y_train, epochs=self.epochs,
                                            class_weight = class_weight)
                y_pred = self.model.predict([np.ones(des_test.shape[0]), des_test[:], body_test[:]])

            
            for i in range(len(test_index)):
                index = test_index[i]
                cw_y[index] = y_pred[i]
            
        return cw_y
    
    ''' Smote + Class Weight '''
    def smote_weight(self):
        smcw_y = dict()
        
        for train_index, test_index in self.kfold.split(self.X):
            print("TEST:", test_index)
            des_train, des_test = self.des[train_index], self.des[test_index]
            body_train, body_test = self.body[train_index], self.body[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            neg, pos = np.bincount(y_train)
            total = neg + pos

            weight_for_0 = (1 / neg)*(total)/2.0 
            weight_for_1 = (1 / pos)*(total)/2.0

            class_weight = {0: weight_for_0, 1: weight_for_1}
    
            sm_des_train, sm_y_train = sm.fit_resample(des_train, y_train)
            sm_body_train, _ = sm.fit_resample(body_train, y_train)
    
            self.model.compile(optimizer= keras.optimizers.Adam(), loss='binary_crossentropy', metrics = ['accuracy'])
        
            if self.TFL is None:
                kf_history = self.model.fit(x = [sm_des_train, sm_body_train], y = sm_y_train, epochs=self.epochs, class_weight = class_weight)
                y_pred = self.model.predict([des_test[:], body_test[:]])
            else:
                kf_history = self.model.fit(x = [np.ones(sm_des_train.shape[0]), sm_des_train, sm_body_train], y = sm_y_train, epochs=self.epochs,
                                            class_weight = class_weight)
                y_pred = self.model.predict([np.ones(des_test.shape[0]), des_test[:], body_test[:]])
            

            for i in range(len(test_index)):
                index = test_index[i]
                smcw_y[index] = y_pred[i]
            
        return smcw_y