import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def frequency_encoding(train_x, test_x, cat_cols):
    for c in cat_cols:
        freq = train_x[c].value_counts()
        train_x[c] = train_x[c].map(freq)
        test_x[c] = test_x[c].map(freq)
    return train_x, test_x


def target_encoding(train_x, train_y, test_x, cat_cols):
    for c in cat_cols:
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # テストデータのカテゴリを置換
        test_x[c] = test_x[c].map(target_mean)

        # 学習データの変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, train_x.shape[0])

        # 学習データを分割
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        for idx_1, idx_2 in kf.split(train_x):
            # out-of-foldで各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            # 変換後の値を一時配列に格納
            tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean)

        # 変換後のデータで元の変数を置換
        train_x[c] = tmp

    return train_x


def test_target_encoding():
    # サンプルデータの作成
    np.random.seed(42)
    train_x = pd.DataFrame({'cat_var': np.random.choice(['A', 'B', 'C'], size=20)})
    train_y = np.random.rand(20) > 0.5  # ランダムなバイナリターゲット
    test_x = pd.DataFrame({'cat_var': np.random.choice(['A', 'B', 'C'], size=10)})
    cat_cols = ['cat_var']

    # target_encoding関数をテスト
    train_x_encoded = target_encoding(train_x.copy(), train_y, test_x.copy(), cat_cols)

    print("Original Train X:\n", train_x)
    print("\nEncoded Train X:\n", train_x_encoded)
    print("\nOriginal Test X:\n", test_x)


def target_encoding_cv(train_x, train_y, test_x, cat_cols):
    # クロスバリデーションのfoldごとにtarget encodingをやり直す
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)):
        # 学習データからバリデーションデータを分ける
        tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # 変数をループしてtarget encoding
        for c in cat_cols:
            # 学習データ全体で各カテゴリにおけるtargetの平均を計算
            data_tmp = pd.DataFrame({c: tr_x[c], 'target': tr_y})
            target_mean = data_tmp.groupby(c)['target'].mean()
            # バリデーションデータのカテゴリを置換
            va_x.loc[:, c] = va_x[c].map(target_mean)

            # 学習データの変換後の値を格納する配列を準備
            tmp = np.repeat(np.nan, tr_x.shape[0])
            kf_encoding = KFold(n_splits=4, shuffle=True, random_state=42)
            for idx_1, idx_2 in kf_encoding.split(tr_x):
                # one-of-fold で各カテゴリにおける目的変数の平均を計算
                target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
                # 変換後の値を一時配列に格納
                tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)

            tr_x.loc[:, c] = tmp

            # 必要に応じてencodeされた特徴量を保存し、あとで読み込めるようにしておく


if __name__ == '__main__':
    test_target_encoding()
