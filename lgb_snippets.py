import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def train_model(X_train, y_train, X_valid, y_valid, params, early_stopping_rounds=50):
    """
    Train a lightgbm model.

    Parameters:
    - X_train: Training features.
    - y_train: Training target.
    - X_valid: Validation features.
    - y_valid: Validation target.
    - params: LGBM parameters.
    - early_stopping_rounds: Early stopping rounds.

    Returns:
    - model: Trained LGBM model.
    """

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True)]
    model = lgb.train(params, train_data, valid_sets=[valid_data], callbacks=callbacks)
    return model


def predict_model(model, X):
    """
    Make predictions using a trained lightgbm model.

    Parameters:
    - model: Trained LGBM model.
    - X: Features to make predictions on.

    Returns:
    - preds: Predictions.
    """
    preds = model.predict(X, num_iteration=model.best_iteration)
    return preds


def evaluate_with_stratifiedkfold(X, y, params, n_splits=3, early_stopping_rounds=50):
    """
    Evaluate model using StratifiedKFold cross-validation.

    Parameters:
    - X: Features.
    - y: Target.
    - params: LGBM parameters.
    - n_splits: Number of K folds.
    - early_stopping_rounds: Early stopping rounds for LGBM.

    Returns:
    - fold_scores: Accuracy scores for each fold.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = train_model(X_train, y_train, X_valid, y_valid, params, early_stopping_rounds=early_stopping_rounds)
        preds = predict_model(model, X_valid)
        preds_class = np.argmax(np.round(preds).astype(int), axis=1)
        accuracy = accuracy_score(y_valid, preds_class)
        fold_scores.append(accuracy)
    return fold_scores


if __name__ == '__main__':
    # サンプルデータセットのロード
    data = load_iris()
    X = data.data
    y = data.target

    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 3,
        # 'n_estimators': 1000,
        'learning_rate': 0.05,
        'verbose': -1
    }
    scores = evaluate_with_stratifiedkfold(X, y, params)
    mean_score = sum(scores) / len(scores)
    print(f"Mean accuracy: {mean_score:.4f}")
