import pandas as pd


def calculate_moving_average(data, window_size):
    """
    Calculate the moving average for the given data.

    :param data: DataFrame or Series. Time series data.
    :param window_size: int. The window size for calculating the moving average.
    :return: DataFrame or Series with moving averages.
    """
    return data.rolling(window=window_size).mean()

def create_lag_features(data, lag_sizes):
    """
    Create lag features for time series data.

    :param data: DataFrame or Series. Time series data.
    :param lag_sizes: list of int. The list of lag sizes to create lag features.
    :return: DataFrame with lag features.
    """
    lagged_data = pd.DataFrame(index=data.index)
    for lag in lag_sizes:
        lagged_data[f'lag_{lag}'] = data.shift(lag)
    return lagged_data

