import unittest

import numpy as np
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

class TestTimeSeriesSplitter(unittest.TestCase):

    def test_time_series_cv_splits(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        n_splits = 3
        splits = time_series_cv_splits(data, n_splits=n_splits)

        self.assertEqual(len(splits), n_splits)
        for train_index, test_index in splits:
            # Ensure test indices are after train indices
            self.assertTrue(np.all(train_index[-1] < test_index[0]))

if __name__ == '__main__':
    unittest.main()
