import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.preprocessing.scaler.data_scaling import (min_max_scale_combined,
                                                   min_max_scale_datasets,
                                                   standardize_combined,
                                                   standardize_datasets)


class TestScalingMethods(unittest.TestCase):

    def setUp(self):
        # テスト用のサンプルデータフレームを初期化
        self.train_df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1]})
        self.test_df = pd.DataFrame({'A': [6, 7, 8, 9, 10], 'B': [10, 9, 8, 7, 6]})

    def test_standardize_datasets(self):
        # standardize_datasets 関数が指定した列を正しく標準化するかをチェック
        columns = ['A', 'B']
        std_train, std_test = standardize_datasets(self.train_df, self.test_df, columns)
        self.assertTrue(np.isclose(std_train[columns].mean(), 0, atol=1e-6).all())

    def test_min_max_scale_datasets(self):
        # min_max_scale_datasets 関数が Min-Max スケーリングを正しく適用するかを検証
        mm_train, mm_test = min_max_scale_datasets(self.train_df, self.test_df, ['A', 'B'])
        self.assertEqual(mm_train['A'].min(), 0)
        self.assertEqual(mm_train['A'].max(), 1)
        self.assertEqual(mm_train['B'].min(), 0)
        self.assertEqual(mm_train['B'].max(), 1)

    def test_min_max_scale_combined(self):
        combined_train, combined_test = min_max_scale_combined(self.train_df, self.test_df, ['A', 'B'])
        # 結合されたデータセット全体の最小値と最大値の確認
        combined_min = min(combined_train['A'].min(), combined_train['B'].min(),
                        combined_test['A'].min(), combined_test['B'].min())
        combined_max = max(combined_train['A'].max(), combined_train['B'].max(),
                        combined_test['A'].max(), combined_test['B'].max())
        self.assertEqual(combined_min, 0)
        self.assertEqual(combined_max, 1)

    def test_standardize_combined(self):
        # standardize_combined 関数のテスト
        columns = ['A', 'B']
        comb_train, comb_test = standardize_combined(self.train_df, self.test_df, columns)
        # 結合されたデータセット全体の平均と標準偏差の確認
        combined = pd.concat([comb_train, comb_test], ignore_index=True)
        self.assertTrue(np.isclose(combined[columns].mean(), 0, atol=1e-6).all())
        self.assertTrue(np.isclose(combined[columns].std(ddof=0), 1, atol=1e-6).all())


if __name__ == '__main__':
    unittest.main()