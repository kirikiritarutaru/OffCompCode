import sys
import unittest
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.preprocessing.feature_engineering.time_series_features import calculate_moving_average, create_lag_features


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
        expected_result = pd.DataFrame({"lag_1": [None, 1, 2, 3, 4], "lag_2": [None, None, 1, 2, 3]})

        lag_features = create_lag_features(data, lag_sizes)
        pd.testing.assert_frame_equal(lag_features, expected_result)


if __name__ == "__main__":
    unittest.main()
