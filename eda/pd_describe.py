import pandas as pd


def pd_describe(data: pd.DataFrame):
    print(data.describe())
    print(data.dtypes)
