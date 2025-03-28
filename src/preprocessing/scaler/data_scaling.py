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
    scaled_train = combined.iloc[: len(train_df), :]
    scaled_test = combined.iloc[len(train_df) :, :]
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
    scaled_train = combined.iloc[: len(train_df), :]
    scaled_test = combined.iloc[len(train_df) :, :]
    return scaled_train, scaled_test
