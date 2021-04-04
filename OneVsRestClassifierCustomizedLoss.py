import numpy as np
from joblib import Parallel, delayed
from sklearn.multiclass import OneVsRestClassifier, _ConstantPredictor
from sklearn import clone
from sklearn.preprocessing import LabelBinarizer
from scipy import special


class OneVsRestClassifierCustomizedLoss(OneVsRestClassifier):

    def __init__(self, estimator, loss):
        self.loss = loss
        super().__init__(estimator)

    def fit(self, X, y, **fit_params):

        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)

        if 'eval_set' in fit_params:
            # use eval_set for early stopping
            X_val, y_val = fit_params['eval_set'][0]
            Y_val = self.label_binarizer_.transform(y_val)
            Y_val = Y_val.tocsc()
            columns_val = (col.toarray().ravel() for col in Y_val.T)
            self.estimators_ = Parallel(n_jobs=None)(delayed(self._fit_binary)
                                                     (self.estimator, X, column, X_val, column_val, **fit_params) for
                                                     i, (column, column_val) in
                                                     enumerate(zip(columns, columns_val)))
        else:
            # eval set not available
            self.estimators_ = Parallel(n_jobs=None)(delayed(self._fit_binary)
                                                     (self.estimator, X, column, None, None, **fit_params) for i, column
                                                     in enumerate(columns))

        return self

    def _fit_binary(self, estimator, X, y, X_val, y_val, **fit_params):
        unique_y = np.unique(y)
        if len(unique_y) == 1:
            estimator = _ConstantPredictor().fit(X, unique_y)
        else:
            estimator = clone(estimator)
            init_score = np.zeros_like(y)
            init_score_value = self.loss.init_score(y)
            init_score.fill(init_score_value)
            if 'eval_set' in fit_params and 'eval_metric' in fit_params:
                estimator.fit(X, y,
                              init_score=init_score,
                              early_stopping_rounds=10,
                              eval_set=[(X_val, y_val)],
                              eval_metric=fit_params['eval_metric'],
                              verbose=False)
            else:
                estimator.fit(X, y,
                              init_score=init_score,
                              verbose=False)

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
