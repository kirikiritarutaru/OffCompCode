import itertools

import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def train_model(X_train, y_train, X_valid, y_valid, params, early_stopping_rounds=50):
    """
    Train an XGBoost model with early stopping.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        early_stopping_rounds=early_stopping_rounds,
        evals=evals,
        verbose_eval=False
    )
    return model


def predict_model(model, X_test):
    """
    Predict with the trained model.
    """
    dtest = xgb.DMatrix(X_test)
    preds = model.predict(dtest)
    return preds


def stratified_kfold_xgboost(X, y, params, n_splits=5, early_stopping_rounds=50):
    """
    Perform Stratified K-Fold cross validation with XGBoost.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        model = train_model(X_train, y_train, X_valid, y_valid, params, early_stopping_rounds)
        predictions = predict_model(model, X_valid)
        accuracy = accuracy_score(y_valid, np.round(predictions))
        fold_accuracies.append(accuracy)
        print(f"Fold {fold} Accuracy: {accuracy}")

    print(f"\nMean Accuracy over {n_splits} folds: {np.mean(fold_accuracies)}")


def stratified_kfold_xgb_params_search(X, y, param_grid, n_splits=5, early_stopping_rounds=50):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        best_score = float("inf")
        best_params = None

        # パラメータの組み合わせを繰り返します
        for params in param_grid:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            # 学習
            evals_result = {}
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=1000,
                early_stopping_rounds=early_stopping_rounds,
                evals=[(dtest, "eval")],
                evals_result=evals_result,
                verbose_eval=False
            )

            # 最良のスコアを更新
            final_score = evals_result['eval'][params['eval_metric']][model.best_iteration]
            if final_score < best_score:
                best_score = final_score
                best_params = params
                print(f"New best score: {best_score}, Params: {best_params}")

        # 最良のパラメータを表示
        print(f"Best score: {best_score}, Best Params: {best_params}")
        return best_params


if __name__ == '__main__':
    # 2値分類の場合
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 42
    }

    # 回帰の場合
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42
    }

    # 多クラス分類の場合
    xgb_params = {
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'num_class': 3,
        'seed': 42,
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    max_depth_options = [3, 6, 9]
    eta_options = [0.01, 0.05, 0.1]
    # パラメータグリッドの例
    param_grid = [
        {
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',
            'seed': 42,
            'num_class': 3,
            'eta': eta,
            'max_depth': depth,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        for depth, eta in itertools.product(max_depth_options, eta_options)
    ]

    data = load_iris()
    X = data.data
    y = data.target
    # stratified_kfold_xgboost(X, y, xgb_params, n_splits=3)
    xgb_params = stratified_kfold_xgb_params_search(X, y, param_grid, n_splits=3)
    stratified_kfold_xgboost(X, y, xgb_params, n_splits=3)
