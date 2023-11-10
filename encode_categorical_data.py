import numpy as np
import pandas as pd
from sklearn import base
from sklearn.model_selection import KFold


def frequency_encoding(train_x, test_x, cat_cols):
    for c in cat_cols:
        freq = train_x[c].value_counts()
        train_x[c] = train_x[c].map(freq)
        test_x[c] = test_x[c].map(freq)
    return train_x, test_x


def target_encoding(train_x, train_y, test_x, cat_cols):
    for c in cat_cols:
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # テストデータのカテゴリを置換
        test_x[c] = test_x[c].map(target_mean)

        # 学習データの変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, train_x.shape[0])

        # 学習データを分割
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        for idx_1, idx_2 in kf.split(train_x):
            # out-of-foldで各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            # 変換後の値を一時配列に格納
            tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

        # 変換後のデータで元の変数を置換
        train_x[c] = tmp

    return train_x


def test_target_encoding():
    # サンプルデータの作成
    np.random.seed(42)
    train_x = pd.DataFrame({'cat_var': np.random.choice(['A', 'B', 'C'], size=20)})
    train_y = np.random.rand(20) > 0.5  # ランダムなバイナリターゲット
    test_x = pd.DataFrame({'cat_var': np.random.choice(['A', 'B', 'C'], size=10)})

    X = pd.concat([train_x, pd.DataFrame(train_y, columns=['target'])], axis=1)
    targetc = KFoldTargetEncoderTrain('cat_var', 'target')
    new_train = targetc.fit_transform(X)
    print(new_train)

    test_targetc = KFoldTargetEncoderTest(new_train, 'cat_var', 'cat_var_Kfold_Target_Enc')
    new_test = test_targetc.fit_transform(test_x)
    print(new_test)


class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):
    """
    参考notebook：
    https://www.kaggle.com/code/anuragbantu/target-encoding-beginner-s-guide
    """

    def __init__(self, colnames, targetName, n_fold=5, verbosity=True, discardOriginal_col=False):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert (isinstance(self.targetName, str))
        assert (isinstance(self.colnames, str))
        assert (self.colnames in X.columns)
        assert (self.targetName in X.columns)
        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=2019)
        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan
        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())
            X[col_mean_name].fillna(mean_of_target, inplace=True)
        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and , {} is {}.'.format(col_mean_name,
                  self.targetName, np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X


class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    """
    参考notebook：
    https://www.kaggle.com/code/anuragbantu/target-encoding-beginner-s-guide
    """

    def __init__(self, train, colNames, encodedName):
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mean = self.train[[self.colNames, self.encodedName]].groupby(self.colNames).mean().reset_index()

        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]
        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd})
        return X


if __name__ == '__main__':
    test_target_encoding()
