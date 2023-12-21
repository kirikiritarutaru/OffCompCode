import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def eda():
    # データセットの読み込み
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    print('df.head: ')
    print(df.head())

    print(f'df.shape: {df.shape}')

    print(f'欠損値：')
    print(df.isnull().sum())

    print('データ型')
    print(df.info())


    print('数値の統計情報')
    print(df.describe().T)

    # 1変数EDA
    print(f'住宅価格の統計情報')
    print(df['MEDV'].describe())

    df['MEDV'].hist(bins=30)
    plt.title('MEDV hist')

    # 2変数EDA
    # 相関係数
    plt.figure(figsize=(12, 10))
    df_corr = df.corr()
    sns.heatmap(df_corr, vmax=1, vmin=-1, center=0, annot=True, cmap = 'Blues')

    # 散布図
    num_cols = ['LSTAT', 'RM', 'MEDV']
    sns.pairplot(df[num_cols], height=2.5)

    plt.show()





if __name__ == '__main__':
    eda()