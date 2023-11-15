import numpy as np
import pandas as pd


def remove_constant_and_duplicate_columns(df_train, df_test=None):
    """
    Removes constant and duplicated columns from the training dataset.
    If a testing dataset is provided, it will remove the same columns from it.

    Parameters:
    - df_train: pd.DataFrame, training dataset
    - df_test: pd.DataFrame or None, testing dataset (optional)

    Returns:
    - df_train_cleaned: pd.DataFrame, training dataset with constant and duplicated columns removed
    - df_test_cleaned: pd.DataFrame or None, testing dataset with the same columns removed (if provided)
    """
    # Remove constant columns
    remove_constant = [col for col in df_train.columns if df_train[col].std() == 0]
    df_train_cleaned = df_train.drop(columns=remove_constant)

    if df_test is not None:
        df_test_cleaned = df_test.drop(columns=remove_constant)
    else:
        df_test_cleaned = None

    # Remove duplicated columns
    remove_duplicates = []
    columns = df_train_cleaned.columns

    for i in range(len(columns) - 1):
        col1_values = df_train_cleaned[columns[i]].values
        for j in range(i + 1, len(columns)):
            if np.array_equal(col1_values, df_train_cleaned[columns[j]].values):
                remove_duplicates.append(columns[j])

    df_train_cleaned = df_train_cleaned.drop(columns=remove_duplicates)

    if df_test_cleaned is not None:
        df_test_cleaned = df_test_cleaned.drop(columns=remove_duplicates)

    return df_train_cleaned, df_test_cleaned


def test_remove_constant_and_duplicate_columns():
    # テスト用のデータフレームを生成
    df_train = pd.DataFrame({
        'A': [1, 1, 1, 1],  # 定数列
        'B': [1, 2, 3, 4],  # 一意の列
        'C': [1, 2, 3, 4],  # 'B'列と重複
        'D': [0, 0, 0, 0],  # 定数列
        'E': [5, 6, 7, 8]   # 一意の列
    })

    df_test = pd.DataFrame({
        'A': [1, 1, 1, 1],
        'B': [4, 5, 6, 7],
        'C': [4, 5, 6, 7],
        'D': [0, 0, 0, 0],
        'E': [8, 9, 10, 11]
    })

    # remove_constant_and_duplicate_columns 関数をテスト
    df_train_cleaned, df_test_cleaned = remove_constant_and_duplicate_columns(df_train, df_test)

    # 結果の検証
    assert 'A' not in df_train_cleaned.columns, "Column 'A' should be removed."
    assert 'C' not in df_train_cleaned.columns, "Column 'C' should be removed."
    assert 'D' not in df_train_cleaned.columns, "Column 'D' should be removed."
    assert 'B' in df_train_cleaned.columns and 'E' in df_train_cleaned.columns, "Columns 'B' and 'E' should not be removed."

    if df_test_cleaned is not None:
        assert 'A' not in df_test_cleaned.columns, "Column 'A' should be removed from test set."
        assert 'C' not in df_test_cleaned.columns, "Column 'C' should be removed from test set."
        assert 'D' not in df_test_cleaned.columns, "Column 'D' should be removed from test set."
        assert 'B' in df_test_cleaned.columns and 'E' in df_test_cleaned.columns, "Columns 'B' and 'E' should not be removed from test set."
    else:
        raise AssertionError("Test set cleaning failed, df_test_cleaned should not be None.")

    print("All tests passed!")


if __name__ == '__main__':
    test_remove_constant_and_duplicate_columns()
