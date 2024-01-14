import io

import lightgbm as lgb
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from graphviz import Source
from sklearn import tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_squared_error, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


# --------------------- Sec 2 ---------------------
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
    # 正の方向に値が大きい特徴量ほど予測値にプラスの影響を与える
    # 負の方向に値が大きい特徴量ほど予測値にマイナスの影響を与える
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


def vis_depth_one_decision_tree():
    # データセットの読み込み
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    # 特徴量と目的変数の設定
    X_train = df.loc[:99, ['RM']]  # 特徴量に100件のRM（平均部屋数）を設定
    y_train = df.loc[:99, 'MEDV']  # 正解値に100件のMEDV（住宅価格）を設定
    print(f'X_train: {X_train[:3]}')
    print(f'y_train: {y_train[:3]}')

    # 深さ1の回帰木モデル
    model = DecisionTreeRegressor(criterion='squared_error', max_depth=1, min_samples_leaf=1, random_state=0)
    model.fit(X_train, y_train)
    print(model.get_params())

    # 学習データの予測と評価
    y_train_pred = model.predict(X_train)
    print('MSE train: %.4f' % (mean_squared_error(y_train, y_train_pred)))

    dot_data = tree.export_graphviz(model, out_file=None, rounded=True, feature_names=X_train.columns, filled=True)
    source = Source(dot_data, format='png')
    png_bytes = source.pipe(format='png')
    sio = io.BytesIO(png_bytes)

    img = mpimg.imread(sio, format='png')
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.axis('off')

    # データと予測値の可視化
    plt.figure(figsize=(8, 4))
    X = X_train.values.flatten()  # numpy配列に変換し、1次元配列に変換
    y = y_train.values  # numpy配列に変換

    # Xの最小値から最大値まで0.01刻みのX_pltを作成し、2次元配列に変換
    X_plt = pd.DataFrame(np.arange(X.min(), X.max(), 0.01)[:, np.newaxis], columns=X_train.columns)
    y_pred = model.predict(X_plt)  # 住宅価格を予測

    # 学習データ(平均部屋数と住宅価格)の散布図と予測値のプロット
    plt.scatter(X, y, color='blue', label='data')
    plt.plot(X_plt, y_pred, color='red', label='DecisionTreeRegressor')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.xlabel('average number of rooms [RM]')
    plt.title('Boston house-prices')
    plt.legend(loc='upper right')

    # データのソート
    X_train = X_train.sort_values('RM')  # 特徴量RMの分割点計算前にソート
    y_train = y_train[X_train.index]  # 正解もソート

    X_train = X_train.values.flatten()  # numpy化して2次元配列→1次元配列
    y_train = y_train.values  # numpy化
    print(X_train[:10])
    print(y_train[:10])

    # 分割点の計算
    index = []
    loss = []
    # 分割点ごとの予測値,SSE,MSEを計算
    # 特徴量RMでの分割点を探索する過程を可視化している
    # 1. 特徴量RMを値の大きさでソート
    # 2. ソートされた特徴量RMの各インデックスで分割し、右の葉と左の葉をつくる
    # 3. 右の葉と左の葉にそれぞれ含まれているデータの平均値を予測値として、目的値との誤差平方和（SSE）を算出する
    #     SSEの算出方法：「右の葉に含まれるレコードの目的変数の平均値とそれぞれのレコードの目的変数との誤差平方和」＋「左の葉に含まれる…」
    # 4. SSEが最も低くなるインデックスのデータを分割点とする
    for i in range(1, len(X_train)):
        X_left = np.array(X_train[:i])
        X_right = np.array(X_train[i:])
        y_left = np.array(y_train[:i])
        y_right = np.array(y_train[i:])
        # 分割点のインデックス
        print('*****')
        print('index', i)
        index.append(i)
        # 左右の分割
        print('X_left:', X_left)
        print('X_right:', X_right)
        print('')
        # 予測値の計算
        print('y_pred_left:', np.mean(y_left))
        print('y_pred_right:', np.mean(y_right))
        print('')
        # SSEの計算
        y_error_left = y_left - np.mean(y_left)
        y_error_right = y_right - np.mean(y_right)
        SSE = np.sum(y_error_left * y_error_left) + np.sum(y_error_right * y_error_right)
        print('SSE:', SSE)
        loss.append(SSE)
        # MSEの計算
        MSE_left = 1/len(y_left) * np.sum(y_error_left * y_error_left)
        MSE_right = 1/len(y_right) * np.sum(y_error_right * y_error_right)
        print('MSE_left:', MSE_left)
        print('MSE_right:', MSE_right)
        print('')

    # 分割点とSSEの可視化
    X_plt = np.array(index)[:, np.newaxis]  # 1次元配列→2次元配列
    plt.figure(figsize=(10, 4))  # プロットのサイズ指定
    plt.plot(X_plt, loss)
    plt.xlabel('Split Point index of feature RM')
    plt.ylabel('SSE')
    plt.title('SSE vs Split Point index')
    plt.grid()
    plt.show()


def vis_depth_two_decision_tree():
    # データセットの読み込み
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    # 特徴量と目的変数の設定
    X_train = df.loc[:99, ['RM']]  # 特徴量に100件のRM（平均部屋数）を設定
    y_train = df.loc[:99, 'MEDV']  # 正解値に100件のMEDV（住宅価格）を設定
    print(f'X_train: {X_train[:3]}')
    print(f'y_train: {y_train[:3]}')

    # 深さ2の回帰木モデル
    model = DecisionTreeRegressor(criterion='squared_error', max_depth=2, min_samples_leaf=1, ccp_alpha=0, random_state=0)
    model.fit(X_train, y_train)
    print(model.get_params())

    dot_data = tree.export_graphviz(model, out_file=None, rounded=True, feature_names=X_train.columns, filled=True)
    source = Source(dot_data, format='png')
    png_bytes = source.pipe(format='png')
    sio = io.BytesIO(png_bytes)

    img = mpimg.imread(sio, format='png')
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.axis('off')

    # データと予測値の可視化
    plt.figure(figsize=(8, 4))  # プロットのサイズ指定
    X = X_train.values.flatten()  # numpy配列に変換し、1次元配列に変換
    y = y_train.values  # numpy配列に変換

    # Xの最小値から最大値まで0.01刻みのX_pltを作成し、2次元配列に変換
    X_plt = pd.DataFrame(np.arange(X.min(), X.max(), 0.01)[:, np.newaxis], columns=X_train.columns)
    y_pred = model.predict(X_plt)  # 住宅価格を予測

    # 学習データ(平均部屋数と住宅価格)の散布図と予測値のプロット
    plt.scatter(X, y, color='blue', label='data')
    plt.plot(X_plt, y_pred, color='red', label='DecisionTreeRegressor')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.xlabel('average number of rooms [RM]')
    plt.title('Boston house-prices')
    plt.legend(loc='upper right')
    plt.show()


def lightgbm_proc():
    # データセットの読み込み
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    # 特徴量と目的変数の設定
    X = df.drop(['MEDV'], axis=1)
    y = df['MEDV']

    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
    print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)

    # ハイパーパラメータの設定
    lgb_train = lgb.Dataset(X_train, y_train)
    params = {
        'objective': 'mse',
        'num_leaves': 5,
        'seed': 0,
        'verbose': -1,
    }

    # モデルの学習
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=50,
        valid_sets=[lgb_train],
        valid_names=['train'],
        callbacks=[lgb.log_evaluation(10)]
    )

    # 学習データの予測と評価
    y_train_pred = model.predict(X_train)
    print('MSE train: %.2f' % (mean_squared_error(y_train, y_train_pred)))
    print('RMSE train: %.2f' % (mean_squared_error(y_train, y_train_pred) ** 0.5))

    # テストデータの予測と評価
    y_test_pred = model.predict(X_test)
    print('RMSE test: %.2f' % (mean_squared_error(y_test, y_test_pred) ** 0.5))

    # 特徴量の重要度の可視化
    importances = model.feature_importance(importance_type='gain')  # 特徴量の重要度
    indices = np.argsort(importances)[::-1]  # 特徴量の重要度を降順にソート

    plt.figure(figsize=(16, 8))  # プロットのサイズ指定
    plt.title('Feature Importance')  # プロットのタイトルを作成
    plt.bar(range(len(indices)), importances[indices])  # 棒グラフを追加
    plt.xticks(range(len(indices)), X.columns[indices], rotation=90)  # X軸に特徴量の名前を追加

    # 木の可視化
    lgb.plot_tree(model, tree_index=0, figsize=(12, 6))
    lgb.plot_tree(model, tree_index=1, figsize=(12, 6))

    explainer = shap.TreeExplainer(
        model=model,
        feature_pertubation='tree_path_dependent'
    )
    # SHAP値の計算
    shap_values = explainer(X_test)

    # 15件目のSHAP値の可視化
    shap.plots.waterfall(shap_values[14])

    # 特徴量重要度の可視化
    shap.plots.bar(shap_values=shap_values)

    plt.show()  # プロットを表示


def lightgbm_predict_use_rm():
    # データセットの読み込み
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    # 特徴量と目的変数の設定
    X_train = df.loc[:99, ['RM']]  # 特徴量に100件のRM（平均部屋数）を設定
    y_train = df.loc[:99, 'MEDV']  # 正解値に100件のMEDV（住宅価格）を設定
    # print('X_train:', X_train[:3])
    # print('y_train:', y_train[:3])

    lgb_train = lgb.Dataset(X_train, y_train)
    params = {
        'objective': 'mse',
        'metric': 'mse',
        'learning_rate': 0.8,
        'max_depth': 1,
        'min_data_in_leaf': 1,
        'min_data_in_bin': 1,
        'max_bin': 100,
        'seed': 0,
        'verbose': -1,
    }

    # モデルの学習
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1,
        valid_sets=[lgb_train],
        valid_names=['train']
    )

    # 学習データの予測と評価
    y_train_pred = model.predict(X_train)
    print('MSE train: %.2f' % (mean_squared_error(y_train, y_train_pred)))

    # 木の可視化
    lgb.plot_tree(model, tree_index=0, figsize=(12, 6))

    # データと予測値の可視化
    plt.figure(figsize=(16, 8))  # プロットのサイズ指定
    X = X_train.values.flatten()  # numpy配列に変換し、1次元配列に変換
    y = y_train.values  # numpy配列に変換

    # Xの最小値から最大値まで0.01刻みのX_pltを作成し、2次元配列に変換
    X_plt = pd.DataFrame(np.arange(X.min(), X.max(), 0.01)[:, np.newaxis], columns=X_train.columns)
    y_pred = model.predict(X_plt)  # 住宅価格を予測

    # 学習データ(平均部屋数と住宅価格)の散布図と予測値のプロット
    plt.scatter(X, y, color='blue', label='data')
    plt.plot(X_plt, y_pred, color='red', label='LightGBM')
    plt.ylabel('Price in $1000s [MEDV]')
    plt.xlabel('average number of rooms [RM]')
    plt.title('Boston house-prices')
    plt.legend(loc='upper right')

    # 初期値
    print('samples:', len(y))  # レコード数
    pred0 = sum(y)/len(y)  # 予測値（平均）
    print('pred0:', pred0)

    # 左葉のレコード
    threshold = 6.793  # 左右に分割する分割点
    X_left = X[X <= threshold]  # 左葉の特徴量
    y_left = y[X <= threshold]  # 左葉の正解値
    print('X_left:', X_left)
    print('')
    print('y_left:', y_left)

    # 左葉の予測値
    print('samples_left:', len(y_left))  # 左葉のレコード数
    residual_left = y_left - pred0  # 残差
    weight_left = sum(residual_left)/len(y_left)  # 重み
    print('weight_left:', weight_left)
    y_pred_left = pred0 + 0.8 * weight_left  # 左葉の予測値
    print('y_pred_left:', y_pred_left)

    # 右葉のレコード
    X_right = X[X > threshold]
    y_right = y[X > threshold]  # 左葉の正解値

    # 右葉の予測値
    print('samples_right:', len(y_right))  # 右葉のレコード数
    residual_right = y_right - pred0  # 残差
    weight_right = sum(residual_right)/len(y_right)  # 重み
    print('weight_right:', weight_right)
    y_pred_right = pred0 + 0.8 * weight_right  # 右葉の予測値
    print('y_pred_right:', y_pred_right)

    # max_bin=20
    X_train = df.loc[:99, ['RM']]  # 特徴量に100件のRM（平均部屋数）を設定
    X_train.hist(bins=20)  # 100件レコードに対してbinが20のヒストグラム

    # max_bin=10
    X_train.hist(bins=10)  # 100件レコードに対してbinが10のヒストグラム

    plt.show()


# --------------------- Sec 3 ---------------------
def sec3_eda():
    # データセットの読み込み
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)
    df.columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
        'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    print(df.head())

    print('データ形状')
    print(df.shape)

    print('欠損値の有無')
    print(df.isnull().sum())

    print('データ型')
    print(df.info())

    print('数値の統計情報')
    print(df.describe().T)

    # 数値のヒストグラム
    plt.rcParams['figure.figsize'] = (10, 6)
    df.hist(bins=20)
    plt.tight_layout()
    plt.show()

    print('カテゴリ変数の統計情報')
    print(df.describe(exclude='number').T)

    # カテゴリ変数のリスト表示
    cat_cols = [
        'workclass', 'education', 'marital-status', 'occupation', 'relationship',
        'race', 'gender', 'native-country', 'income'
    ]
    for col in cat_cols:
        print('%s: %s' % (col, list(df[col].unique())))

    # カテゴリ変数の棒グラフ
    plt.rcParams['figure.figsize'] = (20, 20)
    for i, name in enumerate(cat_cols):
        ax = plt.subplot(5, 2, i+1)
        df[name].value_counts().plot(kind='bar', ax=ax)
    plt.tight_layout()
    plt.show()

    # 半角スペースの削除
    for s in cat_cols:
        df[s] = df[s].str.replace(' ', '')

    print('レコードの絞り込み')
    df = df[df['native-country'].isin(['United-States'])]
    df = df.drop(['native-country'], axis=1)
    df.reset_index(inplace=True, drop=True)
    print(df.shape)

    print('前処理後のincome件数内訳')
    print(df['income'].value_counts())

    print('前処理後のincome件数可視化')
    plt.figure(figsize=(6, 3))
    sns.countplot(x='income', data=df)
    plt.show()

    # 正解ラベルの作成
    df['income'] = df['income'].replace('<=50K', 0)
    df['income'] = df['income'].replace('>50K', 1)

    # 特徴量と目的変数の設定
    X = df.drop(['income'], axis=1)
    y = df['income']

    print('学習データとテストデータに分割')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)
    print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)

    print('学習データとテストデータのラベル件数内訳')
    print(y_train.value_counts())
    print(y_test.value_counts())

    # 予測ラベル0の作成
    y_test_zeros = np.zeros(5834)  # テストデータレコード数の0を作成
    y_test_zeros

    # 予測ラベル0の混同行列
    cm = confusion_matrix(y_test, y_test_zeros)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('pred')
    plt.ylabel('label')
    plt.show()

    # 予測ラベル0の評価指標
    ac_score = accuracy_score(y_test, y_test_zeros)
    pr_score = precision_score(y_test, y_test_zeros, zero_division=0.0)
    rc_score = recall_score(y_test, y_test_zeros)
    f1 = f1_score(y_test, y_test_zeros)

    print('accuracy = %.2f' % (ac_score))
    print('precision = %.2f' % (pr_score))
    print('recall = %.2f' % (rc_score))
    print('F1-score = %.2f' % (f1))

    # 予測ラベル1の作成
    y_test_ones = np.ones(5834)  # テストデータレコード数の1を作成

    # 予測ラベル1の混同行列
    cm = confusion_matrix(y_test, y_test_ones)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('pred')
    plt.ylabel('label')
    plt.show()

    # 予測ラベル1の評価指標
    ac_score = accuracy_score(y_test, y_test_ones)
    pr_score = precision_score(y_test, y_test_ones)
    rc_score = recall_score(y_test, y_test_ones)
    f1 = f1_score(y_test, y_test_ones)

    print('accuracy = %.2f' % (ac_score))
    print('precision = %.2f' % (pr_score))
    print('recall = %.2f' % (rc_score))
    print('F1-score = %.2f' % (f1))


def logistic_reg_proc():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)
    df.columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
        'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    # 前処理
    # 文字列の半角スペース削除
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
    for s in cat_cols:
        df[s] = df[s].str.replace(' ', '')

    # United-Statesのレコードに絞り特徴量native-countryを削除
    df = df[df['native-country'].isin(['United-States'])]
    df = df.drop(['native-country'], axis=1)
    df.reset_index(inplace=True, drop=True)

    # 正解ラベルの数値への置換
    df['income'] = df['income'].replace('<=50K', 0)
    df['income'] = df['income'].replace('>50K', 1)

    print('前処理後のdf.shape')
    print(df.shape)

    # 特徴量と目的変数の設定
    X = df.drop(['income'], axis=1)
    y = df['income']

    # カテゴリ変数
    print('特徴量のカテゴリ変数の情報')
    print(X.describe(exclude='number').T)

    print('特徴量をone-hot encoding')
    X = pd.concat([X, pd.get_dummies(X['workclass'], prefix='workclass', drop_first=True)], axis=1)
    X = pd.concat([X, pd.get_dummies(X['education'], prefix='education', drop_first=True)], axis=1)
    X = pd.concat([X, pd.get_dummies(X['marital-status'], prefix='marital-status', drop_first=True)], axis=1)
    X = pd.concat([X, pd.get_dummies(X['occupation'], prefix='occupation', drop_first=True)], axis=1)
    X = pd.concat([X, pd.get_dummies(X['relationship'], prefix='relationship', drop_first=True)], axis=1)
    X = pd.concat([X, pd.get_dummies(X['race'], prefix='race', drop_first=True)], axis=1)
    X = pd.concat([X, pd.get_dummies(X['gender'], prefix='gender', drop_first=True)], axis=1)
    X = X.drop(['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender'], axis=1)

    # 学習データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)
    print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)

    print('学習データとテストデータのラベル件数内訳')
    print(y_train.value_counts())
    print(y_test.value_counts())

    print('特徴量の標準化')
    scaler = StandardScaler()  # 変換器の作成
    num_cols = X.columns[0:6]  # 数値型の特徴量を取得
    scaler.fit(X_train[num_cols])  # 学習データでの標準化パラメータの計算
    X_train[num_cols] = scaler.transform(X_train[num_cols])  # 学習データの変換
    X_test[num_cols] = scaler.transform(X_test[num_cols])  # テストデータの変換

    # ロジスティック回帰モデル
    model = LogisticRegression(max_iter=100, multi_class='ovr', solver='liblinear', C=0.1, penalty='l1', random_state=0)
    model.fit(X_train, y_train)

    print('ロジスティック回帰モデルのパラメータ')
    print(model.get_params())

    print('テストデータの予測と評価')
    y_test_pred = model.predict(X_test)
    ac_score = accuracy_score(y_test, y_test_pred)
    print('accuracy = %.2f' % (ac_score))

    f1 = f1_score(y_test, y_test_pred)
    print('F1-score = %.2f' % (f1))

    # 混同行列
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('pred')
    plt.ylabel('label')

    # パラメータ
    # print('回帰係数 w = [w1, w2, … , w59]:', model.coef_[0])
    # print('')
    # print('定数項 w0:', model.intercept_)

    # 回帰係数の可視化
    importances = model.coef_[0]  # 回帰係数
    indices = np.argsort(importances)[::-1]  # 回帰係数を降順にソート

    plt.figure(figsize=(20, 8))  # プロットのサイズ指定
    plt.title('Regression coefficient')  # プロットのタイトルを作成
    plt.bar(range(len(indices)), importances[indices])  # 棒グラフを追加
    plt.xticks(range(len(indices)), X.columns[indices], rotation=90)  # X軸に特徴量の名前を追加
    plt.subplots_adjust(bottom=0.4)  # bottomの値を増やして、下部の余白を調整

    # 回帰係数（上位30件）の可視化
    importances = model.coef_[0]  # 回帰係数
    indices = np.argsort(importances)[::-1][:30]  # 回帰係数を降順にソート

    plt.figure(figsize=(10, 6))  # プロットのサイズ指定
    plt.title('Regression coefficient')  # プロットのタイトルを作成
    plt.bar(range(len(indices)), importances[indices])  # 棒グラフを追加
    plt.xticks(range(len(indices)), X.columns[indices], rotation=90)  # X軸に特徴量の名前を追加
    plt.subplots_adjust(bottom=0.5)  # bottomの値を増やして、下部の余白を調整

    # 回帰係数（下位30件）の可視化
    importances = model.coef_[0]  # 回帰係数
    indices = np.argsort(importances)[::-1][-30:]  # 回帰係数を降順にソート

    plt.figure(figsize=(10, 6))  # プロットのサイズ指定
    plt.title('Regression coefficient')  # プロットのタイトルを作成
    plt.bar(range(len(indices)), importances[indices])  # 棒グラフを追加
    plt.xticks(range(len(indices)), X.columns[indices], rotation=90)  # X軸に特徴量の名前を追加
    plt.subplots_adjust(bottom=0.5)  # bottomの値を増やして、下部の余白を調整

    plt.show()  # プロットを表示

    print('最後から3件目のクラス0とクラス1の確率')
    print(model.predict_proba(X_test)[-3])
    print('最後から3件目の特徴量 X = [x1, x2, … , x59]:', X_test.values[-3])

    # 最後から3件目 logit = w × X + w0
    logit = sum(np.multiply(model.coef_[0], X_test.values[-3])) + model.intercept_
    print(f'最後から3件目 logit = w × X + w0: {logit}')

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    print('シグモイド関数でlogitから確率に変換')
    print(sigmoid(logit))


if __name__ == '__main__':
    # 2章
    # eda()
    # multi_reg()
    # vis_simple_reg()
    # vis_decision_tree()
    # vis_depth_one_decision_tree()
    # vis_depth_two_decision_tree()
    # lightgbm_proc()
    # lightgbm_predict_use_rm()

    # 3章
    # sec3_eda()
    logistic_reg_proc()
