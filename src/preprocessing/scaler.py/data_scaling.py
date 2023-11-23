import numpy as np
import pandas as pd
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


def min_max_scale_combined(train_df, test_df, columns):
    """
    Combine train and test datasets, and apply Min-Max scaling to specified columns.

    :param train_df: Training DataFrame
    :param test_df: Testing DataFrame
    :param columns: Columns to scale
    :return: Min-Max scaled training and testing DataFrames
    """
    combined = pd.concat([train_df, test_df], ignore_index=True)
    scaler = MinMaxScaler()
    combined[columns] = scaler.fit_transform(combined[columns])
    scaled_train = combined.iloc[:len(train_df), :]
    scaled_test = combined.iloc[len(train_df):, :]
    return scaled_train, scaled_test


def standardize_combined(train_df, test_df, columns):
    """
    Combine train and test datasets, and standardize specified columns.

    :param train_df: Training DataFrame
    :param test_df: Testing DataFrame
    :param columns: Columns to standardize
    :return: Standardized training and testing DataFrames
    """
    combined = pd.concat([train_df, test_df], ignore_index=True)
    scaler = StandardScaler()
    combined[columns] = scaler.fit_transform(combined[columns])
    scaled_train = combined.iloc[:len(train_df), :]
    scaled_test = combined.iloc[len(train_df):, :]
    return scaled_train, scaled_test


# ---------------------------------------------------------------------------------------------------------------------
# 関数のテストコード


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
    try:
        test_min_max_scale_datasets()
        test_standardize_datasets()
        print("All tests passed successfully.")
    except AssertionError:
        print("Some tests failed.")
    except Exception as e:
        print(f"An error occurred: {e}")
