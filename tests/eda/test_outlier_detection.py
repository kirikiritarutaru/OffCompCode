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
            'col1': [1, 2, 3, 4, 100],  # 100は異常値
            'col2': [2, 4, 6, 8, 10],
            'col3': [5, 10, 15, 20, 25]
        })

    def test_outliers_detection(self):
        # 異常値検出関数のテスト
        result = detect_outliers_with_iqr(self.df)
        self.assertTrue(result.at[4, 'col1'] == 100)  # col1の異常値をチェック
        self.assertTrue(pd.isna(result.at[3, 'col1']))  # 非異常値がNaNであることを確認

    def test_no_outliers(self):
        # 異常値がないケースのテスト
        df_no_outliers = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        result = detect_outliers_with_iqr(df_no_outliers)
        self.assertTrue(result['col1'].isnull().all())  # すべての値がNoneであることを確認

# テストの実行
if __name__ == '__main__':
    unittest.main()
