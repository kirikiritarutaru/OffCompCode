import pandas as pd


def detect_outliers(df):
    outliers = pd.DataFrame(columns=df.columns)
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            mean = df[col].mean()
            std = df[col].std()
            threshold = std * 3
            outliers[col] = df[col].apply(lambda x: x if abs(x - mean) > threshold else None)
    return outliers

def detect_outliers_with_iqr(df):
    outliers = pd.DataFrame(columns=df.columns)
    for col in df.columns:
        if df[col].dtype == 'float64' or df[col].dtype == 'int64':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = df[col].apply(lambda x: x if x < lower_bound or x > upper_bound else None)
    return outliers
