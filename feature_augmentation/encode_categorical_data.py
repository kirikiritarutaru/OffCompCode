import numpy as np
import pandas as pd
from sklearn import base
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


def one_hot_encode(df, columns, binary_encode=False):
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        one_hot = pd.get_dummies(df[column], prefix=column)
        if binary_encode:
            one_hot = one_hot.astype(int)
        df = df.drop(column, axis=1).join(one_hot)
    return df


def frequency_encode(df, columns: list):
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        frequency = df[column].value_counts(normalize=True)
        df[column] = df[column].map(frequency)
    return df


def label_encode(df, columns):
    le = LabelEncoder()
    df[columns] = df[columns].apply(lambda col: le.fit_transform(col))
    return df


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


def test_encodings():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'a'], 'C': ['x', 'y', 'x']})
    result = label_encode(df, ['B', 'C'])
    print(result)

    result = one_hot_encode(df, ['B', 'C'], binary_encode=False)
    print('\n', result)

    result = frequency_encode(df, ['B', 'C'])
    print('\n', result)


if __name__ == '__main__':
    # test_target_encoding()
    test_encodings()
