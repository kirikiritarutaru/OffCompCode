import sys
import unittest
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.preprocessing.scaler.power_transforms import (boxcox_transform,
                                                       yeo_johnson_transform)


class TestTransformations(unittest.TestCase):

    def setUp(self):
        self.train_df = pd.DataFrame({
            'positive_feature1': [1, 2, 3, 4, 5],
            'positive_feature2': [10, 20, 30, 40, 50],
            'mixed_feature': [-1, 0, 1, 2, 3]
        })
        self.test_df = pd.DataFrame({
            'positive_feature1': [6, 7, 8, 9, 10],
            'positive_feature2': [60, 70, 80, 90, 100],
            'mixed_feature': [4, 5, 6, 7, 8]
        })

    def test_boxcox_transform_valid(self):
        # Test for valid Box-Cox transformation
        transformed_train, transformed_test = boxcox_transform(
            self.train_df, self.test_df, ['positive_feature1', 'positive_feature2'])
        self.assertIsNotNone(transformed_train)
        self.assertIsNotNone(transformed_test)

    def test_boxcox_transform_invalid(self):
        # Test for invalid Box-Cox transformation (non-positive values)
        with self.assertRaises(ValueError):
            boxcox_transform(self.train_df, self.test_df, ['mixed_feature'])

    def test_yeo_johnson_transform(self):
        # Test for valid Yeo-Johnson transformation
        transformed_train, transformed_test = yeo_johnson_transform(
            self.train_df, self.test_df, ['positive_feature1', 'positive_feature2', 'mixed_feature'])
        self.assertIsNotNone(transformed_train)
        self.assertIsNotNone(transformed_test)


if __name__ == '__main__':
    unittest.main()