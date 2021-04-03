import lightgbm as lgb
from sklearn.multiclass import OneVsRestClassifier, _ConstantPredictor
from scipy import special
from FocalLoss import FocalLoss

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelBinarizer


class MultiClassLightgbmWithFocalLoss(OneVsRestClassifier):

    def __init__(self, n_jobs=None, **kwargs):
        self.fl = FocalLoss(alpha=kwargs['alpha_focal_loss'], gamma=kwargs['gamma_focal_loss'])
        kwargs['objective'] = self.fl.lgb_obj_array
        kwargs['n_jobs'] = n_jobs
        self.params = kwargs

    def fit(self, X, y, **kwargs):

        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)

        # use eval_set for early stopping
        X_val, y_val = kwargs['eval_set']
        Y_val = self.label_binarizer_.transform(y_val)
        Y_val = Y_val.tocsc()
        columns_val = (col.toarray().ravel() for col in Y_val.T)

        # compute the estimators
        self.estimators_ = \
            Parallel(n_jobs=None)(delayed(self._fit_binary)
                                  (X, column, X_val, column_val) for i, (column, column_val) in
                                  enumerate(zip(columns, columns_val)))

        return self

    def _fit_binary(self, X, y, X_val, y_val):
        unique_y = np.unique(y)
        if len(unique_y) == 1:
            estimator = _ConstantPredictor().fit(X, unique_y)
        else:
            fit = lgb.Dataset(X, y, init_score=np.full_like(y, self.fl.init_score(y), dtype=float))
            val = lgb.Dataset(X_val, y_val, init_score=np.full_like(y_val, self.fl.init_score(y_val), dtype=float),
                              reference=fit)
            estimator = lgb.train(params=self.params,
                                  train_set=fit,
                                  num_boost_round=10000,
                                  valid_sets=(fit, val),
                                  valid_names=('fit', 'val'),
                                  early_stopping_rounds=20,
                                  verbose_eval=False,
                                  fobj=self.fl.lgb_obj,
                                  feval=self.fl.lgb_eval)
        return estimator

    def predict(self, X):

        n_samples = X.shape[0]
        maxima = np.empty(n_samples, dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n_samples, dtype=int)

        for i, e in enumerate(self.estimators_):
            margins = e.predict(X, raw_score=True)
            prob = special.expit(margins)
            np.maximum(maxima, prob, out=maxima)
            argmaxima[maxima == prob] = i

        return argmaxima

    def predict_proba(self, X):
        y = np.zeros((X.shape[0], len(self.estimators_)))
        for i, e in enumerate(self.estimators_):
            margins = e.predict(X, raw_score=True)
            y[:, i] = special.expit(margins)
        y /= np.sum(y, axis=1)[:, np.newaxis]
        return y
