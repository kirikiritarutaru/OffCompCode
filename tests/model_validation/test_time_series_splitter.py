
import sys
import unittest
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.model_validation.time_series_splitter import time_series_cv_splits


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