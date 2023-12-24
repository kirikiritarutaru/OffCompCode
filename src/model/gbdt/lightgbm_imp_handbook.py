import io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graphviz import Source
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def eda():
    # データセットの読み込み
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
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
    sns.heatmap(df_corr, vmax=1, vmin=-1, center=0, annot=True, cmap='Blues')

    # 散布図
    num_cols = ['LSTAT', 'RM', 'MEDV']
    sns.pairplot(df[num_cols], height=2.5)

    plt.show()


def multi_reg():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    # 特徴量と目的変数の設定
    X = df.drop(['MEDV'], axis=1)
    y = df['MEDV']
    print(X.head())

    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
    print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)

    scaler = StandardScaler()  # 変換器の作成
    num_cols = X.columns[0:13]  # 全て数値型の特徴量なので全て取得
    scaler.fit(X_train[num_cols])  # 学習データでの標準化パラメータの計算
    X_train[num_cols] = scaler.transform(X_train[num_cols])  # 学習データの変換
    X_test[num_cols] = scaler.transform(X_test[num_cols])  # テストデータの変換

    print(X_train.iloc[:2])  # 標準化された学習データの特徴量

    model = LinearRegression()  # 線形回帰モデル
    model.fit(X_train, y_train)
    print(model.get_params())

    y_test_pred = model.predict(X_test)
    print('RMSE test: %.2f' % (mean_squared_error(y_test, y_test_pred) ** 0.5))

    print(y_test.describe())

    # パラメータ
    print('回帰係数 w = [w1, w2, … , w13]:', model.coef_)
    print('定数項 w0:', model.intercept_)

    # 回帰係数の可視化
    importances = model.coef_  # 回帰係数
    indices = np.argsort(importances)[::-1]  # 回帰係数を降順にソート

    plt.figure(figsize=(8, 4))  # プロットのサイズ指定
    plt.title('Regression coefficient')  # プロットのタイトルを作成
    plt.bar(range(X.shape[1]), importances[indices])  # 棒グラフを追加
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)  # X軸に特徴量の名前を追加
    plt.show()  # プロットを表示


def vis_simple_reg():
    # データセットの読み込み
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    # 特徴量と目的変数の設定
    X_train = df.loc[:99, ['RM']]  # 特徴量にRM（平均部屋数）を設定
    y_train = df.loc[:99, 'MEDV']  # 正解値にMEDV（住宅価格）を設定
    print('X_train:', X_train[:3])
    print('y_train:', y_train[:3])

    model = LinearRegression()  # 線形回帰モデル
    model.fit(X_train, y_train)
    print(model.get_params())

    # 予測値
    model.predict(X_train)

    # パラメータ
    print('傾き w1:', model.coef_[0])
    print('切片 w0:', model.intercept_)

    # データと予測値の可視化
    plt.figure(figsize=(8, 4))  # プロットのサイズ指定
    X = X_train.values.flatten()  # numpy配列に変換し、1次元配列に変換
    y = y_train.values  # numpy配列に変換

    # Xの最小値から最大値まで0.01刻みのX_pltを作成し、2次元配列に変換
    X_plt = pd.DataFrame(np.arange(X.min(), X.max(), 0.01)[:, np.newaxis], columns=X_train.columns)
    y_pred = model.predict(X_plt)  # 住宅価格を予測

    # 学習データ(平均部屋数と住宅価格)の散布図と予測値のプロット
    plt.scatter(X, y, color='blue', label='data')
    plt.plot(X_plt, y_pred, color='red', label='LinearRegression')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.xlabel('average number of rooms [RM]')
    plt.title('Boston house-prices')
    plt.legend(loc='upper right')
    plt.show()


def vis_decision_tree():
    # データセットの読み込み
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    # 特徴量と目的変数の設定
    X = df.drop(['MEDV'], axis=1)
    y = df['MEDV']

    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
    print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)

    model = DecisionTreeRegressor(criterion='squared_error', max_depth=4, min_samples_leaf=10, ccp_alpha=5, random_state=0)  # 深さ4の回帰木モデル
    model.fit(X_train, y_train)
    print(model.get_params())

    # テストデータの予測と評価
    y_test_pred = model.predict(X_test)
    print('RMSE test: %.2f' % (mean_squared_error(y_test, y_test_pred) ** 0.5))

    dot_data = tree.export_graphviz(model, out_file=None, rounded=True, feature_names=X.columns, filled=True)
    source = Source(dot_data, format='png')
    png_bytes = source.pipe(format='png')
    sio = io.BytesIO(png_bytes)

    # Matplotlibで画像を読み込み、表示
    img = mpimg.imread(sio, format='png')
    plt.figure(figsize=(12, 6))  # ここで図のサイズをインチ単位で指定
    plt.imshow(img)
    plt.axis('off')  # 軸をオフに

    # 特徴量の重要度の可視化
    importances = model.feature_importances_  # 特徴量の重要度
    indices = np.argsort(importances)[::-1]  # 特徴量の重要度を降順にソート

    plt.figure(figsize=(16, 8))  # プロットのサイズ指定
    plt.title('Feature Importance')  # プロットのタイトルを作成
    plt.bar(range(len(indices)), importances[indices])  # 棒グラフを追加
    plt.xticks(range(len(indices)), X.columns[indices], rotation=90)  # X軸に特徴量の名前を追加
    plt.show()  # プロットを表示


if __name__ == '__main__':
    # eda()
    # multi_reg()
    # vis_simple_reg()
    vis_decision_tree()
