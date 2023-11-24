from sklearn.model_selection import TimeSeriesSplit


def time_series_cv_splits(data, n_splits=5):
    """
    Generate train/test indices to split time series data in train/test sets.

    :param data: DataFrame or array-like, shape (n_samples, n_features)
    :param n_splits: int, default=5. Number of splitting iterations in the cross-validator.
    :return: list of tuples (train_index, test_index)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_index, test_index in tscv.split(data):
        splits.append((train_index, test_index))
    return splits
