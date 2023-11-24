import sys
import unittest
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.preprocessing.time_series.time_series_data_format_converter import (
    long_to_wide, wide_to_long)


class TestDataFormatConversion(unittest.TestCase):

    def setUp(self):
        self.data_wide = {
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'User1': [10, 15, 20],
            'User2': [25, 30, 35]
        }
        self.df_wide = pd.DataFrame(self.data_wide)
        self.df_wide = self.df_wide.set_index('Date')

        self.data_long = {
            'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03'],
            'User': ['User1', 'User2', 'User1', 'User2', 'User1', 'User2'],
            'Value': [10, 25, 15, 30, 20, 35]
        }
        self.df_long = pd.DataFrame(self.data_long)
        self.df_long = self.df_long.set_index('Date')

    def test_wide_to_long(self):
        converted = wide_to_long(self.df_wide)
        self.assertTrue(isinstance(converted, pd.DataFrame))
        self.assertEqual(len(converted.columns), 2)  # 'User' and 'Value'
        self.assertEqual(len(converted), len(self.df_wide) * len(self.df_wide.columns))

    def test_long_to_wide(self):
        converted = long_to_wide(self.df_long)
        self.assertTrue(isinstance(converted, pd.DataFrame))
        self.assertEqual(len(converted.columns), len(self.df_long['User'].unique()))
        self.assertEqual(len(converted), len(self.df_long.index.unique()))

    def test_conversion_cycle(self):
        converted_to_long = wide_to_long(self.df_wide)
        converted_back_to_wide = long_to_wide(converted_to_long)
        pd.testing.assert_frame_equal(converted_back_to_wide, self.df_wide)

if __name__ == '__main__':
    unittest.main()
