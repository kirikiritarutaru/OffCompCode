import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def detect_outliers_with_iqr(df):
    outliers = pd.DataFrame(columns=df.columns)
    outlier_indices = []

    for col in df.columns:
        if df[col].dtype == "float64" or df[col].dtype == "int64":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 異常値のフラグを立てる
            outlier_flag = df[col].apply(lambda x: x < lower_bound or x > upper_bound)
            outliers[col] = df[col].where(outlier_flag, None)

            # 異常値のインデックスを収集
            outlier_indices.extend(df[col][outlier_flag].index.tolist())

    # Seabornを使ったボックスプロットの描画
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, palette="Set2")
    plt.title("Box Plot for Detecting Outliers with Seaborn")
    plt.show()

    # 重複を除いて異常値のインデックスを返す
    unique_outlier_indices = list(set(outlier_indices))
    return outliers, unique_outlier_indices
