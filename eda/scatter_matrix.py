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

if __name__ == '__main__':
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

    iris_dataframe = pd.DataFrame(X_train, columns=iris.feature_names)
    plot_scatter_matrix(iris_dataframe)
