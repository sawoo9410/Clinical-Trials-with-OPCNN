"""
Kfold_cv.py - K-fold cross-validation with class imbalance handling.

Provides three training strategies for evaluating dual-input models:
  - base(): Standard K-fold CV without any class balancing.
  - weight(): K-fold CV with inverse-frequency class weights.
  - smote_weight(): K-fold CV with SMOTE oversampling + class weights (recommended).

The dataset is split into descriptor (des, first 13 features) and body
(remaining features) modalities for dual-input model training.

Predictions are collected as a dict mapping sample index -> prediction,
enabling proper reconstruction of the full prediction array after K-fold.
"""

from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
import keras
import numpy as np

sm = SMOTE(random_state=202004)


class K_fold:
    """
    K-fold cross-validation handler for dual-input (des + body) models.

    Args:
        X: Full feature matrix, shape (n_samples, n_features).
           First 13 columns = descriptor features, rest = body features.
        y: Target labels, shape (n_samples,). Binary: 0 (fail) or 1 (pass).
        model: Compiled Keras model with dual inputs [des, body].
        epochs: Number of training epochs per fold.
        kfold: Number of folds for cross-validation (e.g., 10).
        tfn: Set to True for TFL models that require an additional bias input
             of ones. Default None (standard dual-input models).
    """
    def __init__(self, X, y, model, epochs, kfold, tfn=None):
        self.X = X
        self.y = y
        self.model = model
        self.epochs = epochs
        self.kfold = KFold(n_splits = kfold, shuffle = True, random_state = 202004)
        self.des = X[:,:13]    # Descriptor features (molecular properties)
        self.body = X[:,13:]   # Body/tissue expression features
        self.TFL = tfn

    def base(self):
        """
        Standard K-fold CV without class balancing.

        No SMOTE oversampling, no class weights. Serves as a baseline
        to compare the effect of class imbalance handling.

        Returns:
            dict: {sample_index: prediction_value} for all test samples.
        """
        base_y = dict()

        for train_index, test_index in self.kfold.split(self.X):
            print("TEST:", test_index)
            des_train, des_test = self.des[train_index], self.des[test_index]
            body_train, body_test = self.body[train_index], self.body[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.model.compile(optimizer= keras.optimizers.Adam(), loss='binary_crossentropy', metrics = ['accuracy'])

            if self.TFL is None:
                # Standard dual-input model: [des, body]
                kf_history = self.model.fit(x = [des_train, body_train], y = y_train, epochs=self.epochs)
                y_pred = self.model.predict([des_test[:], body_test[:]])
            else:
                # TFL model: requires additional bias input of ones
                kf_history = self.model.fit(x = [np.ones(des_train.shape[0]),des_train, body_train], y = y_train, epochs=self.epochs)
                y_pred = self.model.predict([np.ones(des_test.shape[0]), des_test[:], body_test[:]])

            # Store predictions indexed by original sample position
            for i in range(len(test_index)):
                index = test_index[i]
                base_y[index] = y_pred[i]

        return base_y

    def weight(self):
        """
        K-fold CV with inverse-frequency class weights.

        Computes class weights as (total / (2 * class_count)) per fold
        to penalize misclassification of the minority class more heavily.

        Returns:
            dict: {sample_index: prediction_value} for all test samples.
        """
        cw_y = dict()

        for train_index, test_index in self.kfold.split(self.X):
            print("TEST:", test_index)
            des_train, des_test = self.des[train_index], self.des[test_index]
            body_train, body_test = self.body[train_index], self.body[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Inverse-frequency class weights: balanced weighting
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

    def smote_weight(self):
        """
        K-fold CV with SMOTE oversampling + class weights (recommended).

        Applies SMOTE to the training set to synthesize minority class samples,
        AND uses inverse-frequency class weights during training.
        This dual approach provides the strongest class imbalance handling.

        Note: SMOTE is applied separately to des and body features to maintain
        modality-specific distributions (both use the same y for consistency).

        Returns:
            dict: {sample_index: prediction_value} for all test samples.
        """
        smcw_y = dict()

        for train_index, test_index in self.kfold.split(self.X):
            print("TEST:", test_index)
            des_train, des_test = self.des[train_index], self.des[test_index]
            body_train, body_test = self.body[train_index], self.body[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Inverse-frequency class weights
            neg, pos = np.bincount(y_train)
            total = neg + pos

            weight_for_0 = (1 / neg)*(total)/2.0
            weight_for_1 = (1 / pos)*(total)/2.0

            class_weight = {0: weight_for_0, 1: weight_for_1}

            # SMOTE oversampling: applied independently per modality
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
