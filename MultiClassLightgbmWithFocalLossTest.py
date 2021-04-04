from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score
from OneVsRestClassifierCustomizedLoss import *
from FocalLoss import FocalLoss
import lightgbm as lgb

def test_imbalanced_dataset():
    X, y = make_classification(n_classes=3,
                               n_samples=2000,
                               n_features=4,
                               n_informative=3,
                               n_redundant=1,
                               weights=[.01, .02, .97],
                               random_state=42)

    le = preprocessing.LabelEncoder()
    y_label = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_label, test_size=0.30, random_state=42)

    # clf = lgb.LGBMClassifier()

    loss = FocalLoss(alpha=0.75, gamma=3.0)
    loss_fun = lambda y_true, y_pred: (
    loss.grad(y_true, special.expit(y_pred)), loss.hess(y_true, special.expit(y_pred)))
    estimator = lgb.LGBMClassifier(objective=loss_fun)

    clf = OneVsRestClassifierCustomizedLoss(estimator=estimator, loss=loss)

    eval_metric = lambda y_true, y_pred: ('focal_loss', loss(y_true, special.expit(y_pred)).sum(), False)
    fit_params = {'eval_set': [(X_test, y_test)], 'eval_metric': eval_metric}
    clf.fit(X_train, y_train, **fit_params)

    y_pred = clf.predict(X_test)

    print(y_pred)
    print(y_test)

    acc_score = accuracy_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred, average='macro')
    print(acc_score, ' ', rec_score)

    # save the model
    import dill
    dill.dump(clf, open('model', 'wb'))

    with open('model', 'rb') as file:
        clf = dill.load(file)

    y_pred = clf.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred, average='macro')

    print(acc_score, ' ', rec_score)
