import sys
import unittest
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from eda.outlier_detection import detect_outliers_with_iqr


class TestDetectOutliersWithIQR(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 100],
            'col2': [2, 4, 6, 8, 10],
            'col3': [5, 10, 15, 20, 25]
        })

    def test_outliers_detection(self):
        outliers, indices = detect_outliers_with_iqr(self.df)
        self.assertTrue(outliers['col1'].iloc[4] == 100)
        self.assertIn(4, indices)

    def test_no_outliers(self):
        df_no_outliers = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        outliers, indices = detect_outliers_with_iqr(df_no_outliers)
        self.assertTrue(outliers['col1'].isnull().all())
        self.assertEqual(len(indices), 0)

if __name__ == '__main__':
    unittest.main()