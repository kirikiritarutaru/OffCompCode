import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

"""
参考notebook:
https://www.kaggle.com/code/jacoporepossi/tutorial-cross-validation-nested-cv/notebook
"""


def nestedcv(x, y, classifier, cv_outer, cv_inner, p_grid, test_df):
    clfs = []
    oof_preds = []
    oof_targets = []
    test_preds = []
    for pointer, (train_index, test_index) in enumerate(cv_outer.split(x, y)):
        print(f"\nNested CV: {pointer + 1} of {cv_outer.get_n_splits()} outer fold")
        x_train, x_test = x.loc[train_index], x.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        model = RandomizedSearchCV(classifier, param_distributions=p_grid, scoring="roc_auc", cv=cv_inner, n_jobs=-1)
        model.fit(x_train, y_train)

        pred_test = model.predict_proba(x_test)[:, 1]
        auc_test = roc_auc_score(y_test, pred_test)

        print(f"""
        Best set of parameters on validation: {model.best_params_}
        Best AUC on validation             : {model.best_score_:.3f}

        Test AUC                           : {auc_test:.3f}
        """)
        oof_preds.append(pred_test)
        oof_targets.append(y_test)
        clfs.append(model)

        test_preds.append(model.predict_proba(test_df)[:, 1])

    return clfs, oof_preds, oof_targets, test_preds


def nestedcv_multi_class(x, y, classifier, cv_outer, cv_inner, p_grid, test_df):
    clfs = []
    oof_preds = []
    oof_targets = []
    test_preds = []
    for pointer, (train_index, test_index) in enumerate(cv_outer.split(x, y)):
        print(f"\nNested CV: {pointer + 1} of {cv_outer.get_n_splits()} outer fold")
        x_train, x_test = x.loc[train_index], x.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        model = RandomizedSearchCV(
            classifier, param_distributions=p_grid, scoring="roc_auc_ovr", cv=cv_inner, n_jobs=-1
        )
        model.fit(x_train, y_train)

        pred_test = model.predict_proba(x_test)
        auc_test = roc_auc_score(y_test, pred_test, multi_class="ovr", average="macro")  # 多クラス対応のAUC

        print(f"""
        Best set of parameters on validation: {model.best_params_}
        Best AUC on validation             : {model.best_score_:.3f}

        Test AUC                           : {auc_test:.3f}
        """)
        oof_preds.append(pred_test)
        oof_targets.append(y_test)
        clfs.append(model)

        test_preds.append(model.predict_proba(test_df))

    return clfs, oof_preds, oof_targets, test_preds


def nestedcv_example():
    iris = load_iris()
    x = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target, columns=["target"])

    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    p_grid = {"min_samples_split": [2, 4, 6], "max_depth": [10, 20, 30], "max_features": ["sqrt", "log2"]}

    clfs, oof_preds, oof_targets, test_preds = nestedcv_multi_class(
        x, y["target"], classifier, cv_outer, cv_inner, p_grid, x
    )


if __name__ == "__main__":
    nestedcv_example()
