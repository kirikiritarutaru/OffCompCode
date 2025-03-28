from sklearn.preprocessing import PowerTransformer


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
            raise ValueError(
                f"Column '{col}' contains non-positive values, which are not allowed for Box-Cox transformation."
            )

    # Initialize PowerTransformer for Box-Cox transformation
    pt = PowerTransformer(method="box-cox")

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
    pt = PowerTransformer(method="yeo-johnson")

    # Fit the transformer to the training data and transform training and testing data
    train_df_transformed = train_df.copy()
    test_df_transformed = test_df.copy()
    train_df_transformed[columns] = pt.fit_transform(train_df[columns])
    test_df_transformed[columns] = pt.transform(test_df[columns])

    return train_df_transformed, test_df_transformed
