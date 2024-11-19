import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def prediction(inputx, target, optimal_weights):
    examples = inputx.shape[0]
    pred = np.dot(inputx, optimal_weights)
    error = pred - target
    cost = np.sum(error ** 2) / (2 * examples)
    print(f'Cost function is {cost}')


def visulization(cost_history, numberOfIteration):
    plt.xlabel("iteration")
    plt.ylabel("cost_history")
    x = np.arange(0, numberOfIteration+1)
    plt.plot(x, cost_history)
    plt.show()


def Investigation(data):

    sns.pairplot(data, x_vars=['Feat1', 'Feat2', 'Feat3'],
                 y_vars='Target', height=4, aspect=1, kind='scatter')
    plt.show()
