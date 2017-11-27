import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import perceptron as p


def header_for_iris_data():
    return ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'class']


if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df.columns = header_for_iris_data()
    print(df.head())

    iris_types = df.iloc[0:100, 4]

    X = df.iloc[0:150, [0, 2]].values

    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='sentosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='.', label='versicolor')
    plt.scatter(X[100:150, 0], X[100:150, 1],
                 color='green', marker='+', label='virginica')

    plt.xlabel('sepal length')
    plt.ylabel('petal length')
    plt.legend(loc='upper left')
    plt.show()

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    ppn = p.Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='+')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()