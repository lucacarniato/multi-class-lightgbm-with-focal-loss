import lightgbm as lgb
from sklearn.multiclass import OneVsRestClassifier, _ConstantPredictor
from scipy import special
from FocalLoss import FocalLoss

import numpy as np
from sklearn import clone
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelBinarizer


class MultiClassLightgbmWithFocalLoss(OneVsRestClassifier):

    def __init__(self, n_jobs=None, **kwargs):
        self.fl = FocalLoss(alpha=kwargs['alpha_focal_loss'], gamma=kwargs['gamma_focal_loss'])
        kwargs['objective'] = self.fl.lgb_obj_array
        kwargs['n_jobs'] = n_jobs
        self.estimator = lgb.LGBMClassifier(**kwargs)
        super().__init__(self.estimator)

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

        # store the initial scores
        self.init_score_values = []
        self.estimators_ = \
            Parallel(n_jobs=None)(delayed(self._fit_binary)
                                  (self.estimator, X, column, X_val, column_val) for i, (column, column_val) in enumerate(zip(columns, columns_val)))

        return self

    def _fit_binary(self, estimator, X, y, X_val, y_val):
        unique_y = np.unique(y)
        if len(unique_y) == 1:
            estimator = _ConstantPredictor().fit(X, unique_y)
        else:
            estimator = clone(estimator)
            init_score = np.zeros_like(y)
            init_score_value = self.fl.init_score(y)
            init_score.fill(init_score_value)
            self.init_score_values.append(init_score_value)
            estimator.fit(X, y,
                          init_score=init_score,
                          early_stopping_rounds=20,
                          eval_set=[(X_val, y_val)],
                          eval_metric=self.fl.lgb_eval_array,
                          verbose=False)

            fit = lightgbm.Dataset(X, Y,
                                   init_score=np.full_like(train_y_fold, fl.init_score(train_y_fold), dtype=float))
            val = lightgbm.Dataset(test_x_fold, test_y_fold,
                                   init_score=np.full_like(test_y_fold, fl.init_score(test_y_fold), dtype=float),
                                   reference=fit)

            estimator = lightgbm.train(params=params,
                                   train_set=fit,
                                   num_boost_round=10000,
                                   valid_sets=(fit, val),
                                   valid_names=('fit', 'val'),
                                   early_stopping_rounds=20,
                                   verbose_eval=False,
                                   fobj=fl.lgb_obj,
                                   feval=fl.lgb_eval_recall)


        return estimator

    def predict(self, X):

        n_samples = X.shape[0]
        maxima = np.empty(n_samples, dtype=float)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n_samples, dtype=int)

        for i, (e,isv) in enumerate(zip(self.estimators_,self.init_score_values)):
            margins = e.predict(X,raw_score=True)
            prob = special.expit(margins)
            np.maximum(maxima, prob, out=maxima)
            argmaxima[maxima == prob] = i

        return argmaxima

    def predict_proba(self, X):
        y = np.zeros((X.shape[0], len(self.estimators_)))
        for i, (e,isv) in enumerate(zip(self.estimators_,self.init_score_values)):
            margins = e.predict(X, raw_score=True)
            y[:, i] = special.expit(margins)
        y /= np.sum(y, axis=1)[:, np.newaxis]
        return y