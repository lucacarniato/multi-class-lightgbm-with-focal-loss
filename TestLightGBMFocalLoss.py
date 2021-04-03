from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score
from LightgbmFocalLoss import *


def test_with_iris():
    X, y = make_classification(n_classes=3,
                               n_samples=500,
                               n_features=4,
                               n_informative=3,
                               n_redundant=1,
                               weights=[.03, .02, .95],
                               random_state=42)

    le = preprocessing.LabelEncoder()  #
    y_label = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.30, random_state=42)

    model_params = {"num_leaves": 60,
                    "max_depth": -1,
                    "learning_rate": 0.01,
                    "alpha_focal_loss": 0.75,
                    "gamma_focal_loss": 2.0}

    #clf = lgb.LGBMClassifier(n_jobs=3, **model_params)
    #clf.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_test, y_test)])

    clf = MultiClassLightgbmWithFocalLoss(n_jobs=3, **model_params)
    fit_params = {'eval_set': (X_test, y_test)}
    clf.fit(X_train, y_train, **fit_params)
    y_pred = clf.predict(X_test)

    print(y_pred)
    print(y_test)

    acc_score = accuracy_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred, average='macro')
    print(acc_score,' ', rec_score)
