# src/models/xgb_classifier.py

import numpy as np
import pandas as pd
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from scipy.stats import hmean, gmean

#   0%|          | 0/1000 [00:00<?, ?trial/s, best loss=?]/usr/local/lib/python3.10/dist-packages/xgboost/core.py:160: UserWarning: [17:42:03] WARNING: /workspace/src/common/error_msg.cc:58: Falling back to prediction using DMatrix due to mismatched devices. This might lead to higher memory usage and slower performance. XGBoost is running on: cuda:0, while the input data is on: cpu.
# Potential solutions:
# - Use a data structure that matches the device ordinal in the booster.
# - Set the device for booster before call to inplace_predict.

class XGBClassifier:
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.best_params = None
        self.space = {
            'max_depth': hp.choice('max_depth', range(3, 20, 1)),  # Increased upper limit
            'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),  # Expanded range
            'subsample': hp.uniform('subsample', 0.4, 1),  # Slightly expanded range
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),  # Slightly expanded range
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),  # Expanded range and precision
            'n_estimators': hp.choice('n_estimators', range(50, 1500, 50)),  # Expanded range and granularity
            'gamma': hp.quniform('gamma', 0, 10, 0.5),  # Expanded range and precision
            'reg_lambda': hp.uniform('reg_lambda', 0, 5),  # Expanded range
            'reg_alpha': hp.uniform('reg_alpha', 0, 5),  # Expanded range
            'tree_method': 'hist',  # Use 'hist' for histogram-based algorithm
            'device': 'cuda',  # Use GPU with CUDA
            'early_stopping_rounds': hp.choice('early_stopping_rounds', range(10, 101, 10))
        }

    def optimize(self, X_train, y_train, X_val, y_val):
        """
        Optimize the hyperparameters of the XGBoost model using Hyperopt.

        Args:
            X_train, y_train: Training dataset.
            X_val, y_val: Validation dataset for early stopping.
        """
        self.space['scale_pos_weight'] = np.sum(y_train == 0) / np.sum(y_train == 1)

        def objective(params):
            clf = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)

            y_pred = clf.predict(X_val)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            mean = hmean([metric for metric in [sensitivity, specificity, ppv, npv] if metric > 0])

            return {'loss': -np.mean(mean), 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=1000,
                    trials=trials)

        self.best_params = space_eval(self.space, best)
        return self.best_params

    def fit(self, X_train, y_train):
        """
        Fit the XGBoost model on the training dataset.
        """
        if self.best_params is None:
            raise Exception("Model has not been optimized yet. Please run optimize() first.")
        
        self.model = xgb.XGBClassifier(**self.best_params, use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the XGBoost model on the test dataset.
        """
        if self.model is None:
            raise Exception("Model is not trained. Please run fit() first.")

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = np.round(y_pred_proba)

        auroc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)

        metrics = {
            'AUROC': auroc,
            'Accuracy': accuracy,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV': ppv,
            'NPV': npv
        }

        results_df = pd.DataFrame()
        results_df['y_test'] = y_test
        results_df['y_pred'] = y_pred
        results_df['y_pred_proba'] = y_pred_proba

        return metrics, results_df