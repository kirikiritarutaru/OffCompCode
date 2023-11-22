import numpy as np
import pandas as pd
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   StandardScaler)


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


def boxcox_transform(train_df, test_df, columns):
    """
    Apply Box-Cox transformation to specified columns of training and testing data,
    based on training data statistics.

    :param train_df: Training DataFrame
    :param test_df: Testing DataFrame
    :param columns: Columns to apply Box-Cox transformation
    :return: Box-Cox transformed training and testing DataFrames
    """
    # Check if columns contain only positive values in training data
    for col in columns:
        if train_df[col].le(0).any():
            raise ValueError(f"Column '{col}' contains non-positive values, which are not allowed for Box-Cox transformation.")

    # Initialize PowerTransformer for Box-Cox transformation
    pt = PowerTransformer(method='box-cox')

    # Fit the transformer to the training data and transform training and testing data
    train_df_transformed = train_df.copy()
    test_df_transformed = test_df.copy()
    train_df_transformed[columns] = pt.fit_transform(train_df[columns])
    test_df_transformed[columns] = pt.transform(test_df[columns])

    return train_df_transformed, test_df_transformed


def yeo_johnson_transform(train_df, test_df, columns):
    """
    Apply Yeo-Johnson transformation to specified columns of training and testing data,
    based on training data statistics.

    :param train_df: Training DataFrame
    :param test_df: Testing DataFrame
    :param columns: Columns to apply Yeo-Johnson transformation
    :return: Yeo-Johnson transformed training and testing DataFrames
    """
    # Initialize PowerTransformer for Yeo-Johnson transformation
    pt = PowerTransformer(method='yeo-johnson')

    # Fit the transformer to the training data and transform training and testing data
    train_df_transformed = train_df.copy()
    test_df_transformed = test_df.copy()
    train_df_transformed[columns] = pt.fit_transform(train_df[columns])
    test_df_transformed[columns] = pt.transform(test_df[columns])

    return train_df_transformed, test_df_transformed

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


def test_boxcox_transform():
    # Create sample data for testing
    train_df = pd.DataFrame({'positive_feature1': [1, 2, 3, 4, 5],
                             'positive_feature2': [10, 20, 30, 40, 50],
                             'mixed_feature': [-1, 0, 1, 2, 3]})
    test_df = pd.DataFrame({'positive_feature1': [6, 7, 8, 9, 10],
                            'positive_feature2': [60, 70, 80, 90, 100],
                            'mixed_feature': [4, 5, 6, 7, 8]})

    # Test for valid Box-Cox transformation
    try:
        transformed_train, transformed_test = boxcox_transform(train_df, test_df, ['positive_feature1', 'positive_feature2'])
        assert transformed_train is not None and transformed_test is not None
    except Exception as e:
        assert False, f"Box-Cox transformation failed with error: {e}"

    # Test for invalid Box-Cox transformation (non-positive values)
    try:
        boxcox_transform(train_df, test_df, ['mixed_feature'])
        assert False, "Box-Cox transformation should fail for non-positive values"
    except ValueError:
        assert True  # Expected behavior
    except Exception as e:
        assert False, f"Unexpected error for non-positive values: {e}"


def test_yeo_johnson_transform():
    # Create sample data for testing
    train_df = pd.DataFrame({'feature1': [1, 2, 3, 4, 5],
                             'feature2': [10, 20, 30, 40, 50],
                             'negative_feature': [-5, -4, -3, -2, -1]})
    test_df = pd.DataFrame({'feature1': [6, 7, 8, 9, 10],
                            'feature2': [60, 70, 80, 90, 100],
                            'negative_feature': [-10, -9, -8, -7, -6]})

    # Test for valid Yeo-Johnson transformation
    try:
        transformed_train, transformed_test = yeo_johnson_transform(train_df, test_df, ['feature1', 'feature2', 'negative_feature'])
        assert transformed_train is not None and transformed_test is not None
    except Exception as e:
        assert False, f"Yeo-Johnson transformation failed with error: {e}"


if __name__ == '__main__':
    # テストの実行
    try:
        test_min_max_scale_datasets()
        test_standardize_datasets()
        test_boxcox_transform()
        test_yeo_johnson_transform()
        print("All tests passed successfully.")
    except AssertionError:
        print("Some tests failed.")
    except Exception as e:
        print(f"An error occurred: {e}")
