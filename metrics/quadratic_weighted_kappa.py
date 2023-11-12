import time
from typing import Union

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score


def qwk(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    max_rat: int = 3
) -> float:
    y_true_ = np.asarray(y_true)
    y_pred_ = np.asarray(y_pred)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    uniq_class = np.unique(y_true_)
    for i in uniq_class:
        hist1[int(i)] = len(np.argwhere(y_true_ == i))
        hist2[int(i)] = len(np.argwhere(y_pred_ == i))

    numerator = np.square(y_true_ - y_pred_).sum()

    denominator = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            denominator += hist1[i] * hist2[j] * (i - j) * (i - j)

    denominator /= y_true_.shape[0]
    return 1 - numerator / denominator


class OptimizedRounder(object):
    def __init__(
        self,
        n_overall: int = 5,
        n_classwise: int = 5,
        n_classes: int = 7,
        metric: str = "qwk"
    ):
        self.n_overall = n_overall
        self.n_classwise = n_classwise
        self.n_classes = n_classes
        self.coef = [1.0 / n_classes * i for i in range(1, n_classes)]
        self.metric_str = metric
        self.metric = qwk if metric == "qwk" else accuracy_score

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        X_p = np.digitize(X, self.coef)
        if self.metric_str == "qwk":
            ll = -self.metric(y, X_p, self.n_classes - 1)
        else:
            ll = -self.metric(y, X_p)
        return ll

    def fit(self, X: np.ndarray, y: np.ndarray):
        golden1 = 0.618
        golden2 = 1 - golden1
        ab_start = [
            (0.01, 1.0 / self.n_classes + 0.05),
        ]
        for i in range(1, self.n_classes):
            ab_start.append((i * 1.0 / self.n_classes + 0.05,
                             (i + 1) * 1.0 / self.n_classes + 0.05))
        for _ in range(self.n_overall):
            for idx in range(self.n_classes - 1):
                # golden section search
                a, b = ab_start[idx]
                # calc losses
                self.coef[idx] = a
                la = self._loss(X, y)
                self.coef[idx] = b
                lb = self._loss(X, y)
                for it in range(self.n_classwise):
                    # choose value
                    if la > lb:
                        a = b - (b - a) * golden1
                        self.coef[idx] = a
                        la = self._loss(X, y)
                    else:
                        b = b - (b - a) * golden2
                        self.coef[idx] = b
                        lb = self._loss(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_p = np.digitize(X, self.coef)
        return X_p


if __name__ == '__main__':
    # テストデータの生成
    np.random.seed(0)
    y_true = np.random.choice([0, 1, 2, 3], 100, p=[0.1, 0.4, 0.4, 0.1])
    X = y_true + np.random.normal(0, 0.5, 100)  # y_true に基づいて少しノイズを加える

    optr = OptimizedRounder(n_classes=4, metric="qwk")
    optr.fit(X, y_true)
    y_pred = optr.predict(X)

    start = time.time()
    score = qwk(y_true, y_pred, max_rat=3)
    end = time.time()
    print(f"QWKスコア: {score} ({end-start:.6f})")
    start = time.time()
    score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    end = time.time()
    print(f'sklearn qwk: {score} ({end-start:.6f})')

    assert 0 <= score <= 1
