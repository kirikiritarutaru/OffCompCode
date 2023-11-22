import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def standardize_datasets(train_df, test_df, columns):
    """
    Standardize specified columns of train and test datasets using the mean and std of the train dataset.

    :param train_df: Training DataFrame
    :param test_df: Testing DataFrame
    :param columns: List of columns to standardize
    :return: Standardized training and testing DataFrames
    """
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()

    for col in columns:
        scaler.fit(train_df[[col]])
        train_df_scaled[col] = scaler.transform(train_df[[col]])
        test_df_scaled[col] = scaler.transform(test_df[[col]])

    return train_df_scaled, test_df_scaled


def min_max_scale_datasets(train_df, test_df, columns):
    """
    Apply Min-Max scaling to specified columns of train and test datasets using the range of the train dataset.

    :param train_df: Training DataFrame
    :param test_df: Testing DataFrame
    :param columns: List of columns to apply Min-Max scaling
    :return: Min-Max scaled training and testing DataFrames
    """
    scaler = MinMaxScaler()
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()

    for col in columns:
        scaler.fit(train_df[[col]])
        train_df_scaled[col] = scaler.transform(train_df[[col]])
        test_df_scaled[col] = scaler.transform(test_df[[col]])

    return train_df_scaled, test_df_scaled


def apply_boxcox(df, column):
    """
    指定されたDataFrameのカラムにBox-Cox変換を適用します。

    :param df: pandas DataFrame
    :param column: Box-Cox変換を適用するカラムの名前
    :return: Box-Cox変換されたカラムを含むDataFrame
    """
    # 指定されたカラムがDataFrame内に存在するかチェック
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # データが正の値かどうかチェック
    if any(df[column] <= 0):
        raise ValueError(f"Column '{column}' contains non-positive values")

    # Box-Cox変換の実行
    df[column], _ = stats.boxcox(df[column])

    return df


def apply_yeo_johnson(df, column):
    """
    指定されたDataFrameのカラムにYeo-Johnson変換を適用します。

    :param df: pandas DataFrame
    :param column: Yeo-Johnson変換を適用するカラムの名前
    :return: Yeo-Johnson変換されたカラムを含むDataFrame
    """
    # 指定されたカラムがDataFrame内に存在するかチェック
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # Yeo-Johnson変換の実行
    df[column], _ = stats.yeojohnson(df[column])

    return df


# ---------------------------------------------------------------------------------------------------------------------
# 関数のテストコード


def test_apply_boxcox():
    # テスト用のデータフレームを作成
    test_df = pd.DataFrame(
        {'positive_data': [1, 2, 3, 4, 5],
         'mixed_data': [0, -1, 1, 2, 3]}
    )

    # 正のデータに対してBox-Cox変換を実行
    transformed_df = apply_boxcox(test_df, 'positive_data')

    # 変換後のデータの検証
    assert 'positive_data' in transformed_df.columns
    assert all(transformed_df['positive_data'] >= 0)

    # 負のデータが含まれる場合のエラーチェック
    try:
        apply_boxcox(test_df, 'mixed_data')
        assert False  # この行は実行されないはず
    except ValueError:
        assert True  # 正しいエラーが発生したことを確認


def test_apply_yeo_johnson():
    # テスト用のデータフレームを作成
    test_df = pd.DataFrame(
        {'data': [1, 2, 3, 4, 5],
         'negative_data': [-1, -2, -3, -4, -5]}
    )

    # 正のデータに対してYeo-Johnson変換を実行
    transformed_df = apply_yeo_johnson(test_df, 'data')

    # 変換後のデータの検証
    assert 'data' in transformed_df.columns
    assert transformed_df['data'].isnull().sum() == 0

    # 負のデータに対してYeo-Johnson変換を実行（エラーは発生しない）
    transformed_negative_df = apply_yeo_johnson(test_df, 'negative_data')
    assert 'negative_data' in transformed_negative_df.columns
    assert transformed_negative_df['negative_data'].isnull().sum() == 0


def test_standardize_datasets():
    # Create sample data
    train_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
    test_df = pd.DataFrame({'A': [6, 7, 8, 9, 10], 'B': [10, 9, 8, 7, 6]})

    # Test standardization function
    columns = ['A', 'B']
    std_train, std_test = standardize_datasets(train_df, test_df, columns)
    assert np.isclose(std_train[columns].mean(), 0, atol=1e-6).all(), "Mean is not close to 0 in train data"


def test_min_max_scale_datasets():
    # Create sample data
    train_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
    test_df = pd.DataFrame({'A': [6, 7, 8, 9, 10], 'B': [10, 9, 8, 7, 6]})

    # Test Min-Max scaling function
    mm_train, mm_test = min_max_scale_datasets(train_df, test_df, ['A', 'B'])
    assert mm_train['A'].min() == 0 and mm_train['A'].max() == 1
    assert mm_train['B'].min() == 0 and mm_train['B'].max() == 1


if __name__ == '__main__':
    # テストの実行
    test_standardize_datasets()
    try:
        test_apply_boxcox()
        test_apply_yeo_johnson()
        test_min_max_scale_datasets()
        print("All tests passed successfully.")
    except AssertionError:
        print("Some tests failed.")
    except Exception as e:
        print(f"An error occurred: {e}")
