import unittest

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



class TestTimeSeriesFeatures(unittest.TestCase):

    def test_calculate_moving_average(self):
        data = pd.Series([1, 2, 3, 4, 5])
        window_size = 2
        expected_result = pd.Series([None, 1.5, 2.5, 3.5, 4.5])

        moving_avg = calculate_moving_average(data, window_size)
        pd.testing.assert_series_equal(moving_avg, expected_result, check_names=False)

    def test_create_lag_features(self):
        data = pd.Series([1, 2, 3, 4, 5])
        lag_sizes = [1, 2]
        expected_result = pd.DataFrame({
            'lag_1': [None, 1, 2, 3, 4],
            'lag_2': [None, None, 1, 2, 3]
        })

        lag_features = create_lag_features(data, lag_sizes)
        pd.testing.assert_frame_equal(lag_features, expected_result)

if __name__ == '__main__':
    unittest.main()
