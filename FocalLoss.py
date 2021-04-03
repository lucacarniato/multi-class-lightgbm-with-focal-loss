import numpy as np
from scipy import optimize
from scipy import special
from sklearn.metrics import accuracy_score, recall_score

class FocalLoss:

    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        res = optimize.minimize_scalar(lambda p: self(y_true, p).sum(), bounds=(0, 1), method='bounded')
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj_array(self, y_true, y_pred):
        p = special.expit(y_pred)
        return self.grad(y_true, p), self.hess(y_true, p)

    def lgb_obj(self, y_pred, train_data):
        y_true = train_data.get_label()
        return self.lgb_obj_array(y_true, y_pred)

    def lgb_eval_array(self,y_true, y_pred):
        p = special.expit(y_pred)
        is_higher_better = False
        return 'focal_loss', self(y_true, p).sum(),is_higher_better

    def lgb_eval(self, y_pred, train_data):
        y_true = train_data.get_label()
        return self.lgb_eval_array(y_true,y_pred)

    def lgb_eval_accuracy_array(self,y_true, y_pred):
        p = special.expit(y_pred)
        pred_labeles = [1 if v > 0.5 else 0 for v in p]
        score  = accuracy_score(y_true, pred_labeles)
        is_higher_better = True
        return 'focal_loss', score, is_higher_better

    def lgb_eval_recall_array(self,y_true, y_pred):
        p = special.expit(y_pred)
        pred_labeles = [1 if v > 0.5 else 0 for v in p]
        score  = recall_score(y_true, pred_labeles, average='macro')
        is_higher_better = True
        return 'focal_loss', score, is_higher_better


