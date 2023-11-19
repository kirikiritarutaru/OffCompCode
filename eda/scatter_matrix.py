import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def plot_scatter_matrix(data):
    pd.plotting.scatter_matrix(
        data, c=y_train, marker='o', hist_kwds={'bins': 20},
        s=60, alpha=0.8, diagonal='hist', figsize=(12, 12)
    )
    plt.show()


def plot_histograms(df, bins=20):
    """
    この関数はpd.DataFrameを受け取り、各カラムのヒストグラムを描画します。
    """
    num_columns = df.shape[1]
    fig, axes = plt.subplots(num_columns, 1, figsize=(8, num_columns * 4))

    if num_columns == 1:
        axes = [axes]

    for ax, col in zip(axes, df.columns):
        df[col].hist(ax=ax, bins=bins)
        ax.set_title(f'Histogram of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

    iris_dataframe = pd.DataFrame(X_train, columns=iris.feature_names)
    plot_histograms(iris_dataframe)
    # plot_scatter_matrix(iris_dataframe)
