import numpy as np
from load_and_processing import *


def normal_equation(input_data, target):
    Optimal_weights = np.dot(np.dot(np.linalg.inv(
        np.dot(input_data.T, input_data)), input_data.T), target)
    print(Optimal_weights)
    return Optimal_weights


def Simple_Linear_Regression_Derivation(input_data, target):

    W1 = ((input_data.sum() * target.sum()) - (input_data.shape[0] * np.dot(input_data.T, target)))/(
        (input_data.sum())**2 - (input_data.shape[0] * np.dot(input_data.T, input_data)))
    W0 = target.mean()-(W1*input_data.mean())
    print(W0, W1)
