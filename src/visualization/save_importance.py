from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_importances(importances_: pd.DataFrame):
    """
    LightGBM, XGBoostなどのfeature importanceを綺麗に描画する関数
    参考：
    https://qiita.com/kaggle_grandmaster-arai-san/items/d59b2fb7142ec7e270a5

    使い方：
        importances = pd.DataFrame()

        for fold_, (trn_idx, val_idx) in enumerate(kf.split(full)):
            # (中略)

            imp_df = pd.DataFrame()
            imp_df['feature'] = full.columns
            imp_df['gain'] = clf.feature_importances_
            imp_df['fold'] = fold_ + 1
            importances = pd.concat([importances, imp_df], axis=0, sort=False)

        save_importances(importances)

    """
    mean_gain = importances_[["gain", "feature"]].groupby("feature").mean()
    importances_["mean_gain"] = importances_["feature"].map(mean_gain["gain"])
    plt.figure(figsize=(8, 12))
    sns.barplot(
        x="gain",
        y="feature",
        data=importances_.sort_values("mean_gain", ascending=False)[:300],
        hue="gain",
        palette="hls",
    )
    plt.tight_layout()
    plt.savefig("importances.png")


if __name__ == "__main__":
    # テストデータの生成
    num_features = 50
    data = {"gain": np.random.rand(num_features), "feature": [f"feature{i}" for i in range(1, num_features + 1)]}
    importances_ = pd.DataFrame(data)

    # テスト関数の実行
    save_importances(importances_)

    # ファイルの生成を確認
    file_path = Path("importances.png")
    print("File created:", file_path.exists())
