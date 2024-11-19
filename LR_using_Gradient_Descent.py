import numpy as np
import matplotlib.pyplot as plt
from load_and_processing import *


def gradient_checking(f, df, weights):
    eps = 1e-4
    for indx in range(len(weights)):
        weights[indx] -= eps
        cost1 = f(weights)
        weights[indx] += 2*eps
        cost2 = f(weights)
        weights[indx] -= eps
        numerically_derivative = (cost2-cost1)/(2*eps)
        analytic_derivative = df(weights)[indx]
        if (np.isclose(numerically_derivative, analytic_derivative, atol=0.001) == False):
            print(numerically_derivative, analytic_derivative)
            return False
    return True


def linear_regression_using_gradient_descent(inputx, target, step_size=0.01, percesion=0.0001, max_iterative=10000):
    # example
    examples, features = inputx.shape
    inti_waight = np.array([0, 0.6338432,  0.20894728, 0.00150253])

    def f(weight):
        # prediction = inputx@weight
        prediction = np.dot(inputx, weight)
        error = prediction-target
        error_square = np.dot(error.T, error)
        cost = error_square/(2*examples)
        return cost

    def f_derivative(weight):
        # prediction = inputx@weight
        prediction = np.dot(inputx, weight)
        error = prediction-target
        derivative = np.dot(inputx.T, error)
        gradient = derivative / examples
        return gradient
    old_point = inti_waight.copy()
    new_point = old_point*2+5
    visited_points = [old_point.copy()]
    cost_history = [f(old_point)]
    optimal_weights = 0
    iterative = 0
    cost = 0
    # print("is               ", gradient_checking(f, f_derivative, inti_waight))
    while (iterative < max_iterative) and (np.linalg.norm(new_point-old_point) > percesion):
        new_point = old_point.copy()
        cost = f(old_point)
        gradient = f_derivative(old_point)
        visited_points.append(old_point.copy())
        cost_history.append(f(old_point))
        old_point -= gradient*step_size

        iterative += 1
    optimal_weights = old_point
    print(optimal_weights)

    return optimal_weights, visited_points, cost_history, cost, iterative


def Optimizing_the_hyperparameters(inputx, target):
    minimum_error = float('inf')
    minimum_iteration = 10000
    step_size_and_Precision_with_minimum_error = [0.1, 0.01]
    Step_sizes = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0000001]
    Precision = [0.01, 0.001, 0.0001, 0.00001]
    for i in Step_sizes:
        for j in Precision:
            for q in range(3):
                optimal_weights, visited_points, cost_history, cost, iterative = linear_regression_using_gradient_descent(
                    inputx, target, Step_sizes, Precision, 10000)
                if (cost < minimum_error):
                    minimum_error = cost
                    step_size_and_Precision_with_minimum_error[i, j]
                if (iterative < minimum_iteration):
                    minimum_iteration = iterative
        print(minimum_error, minimum_iteration,
              step_size_and_Precision_with_minimum_error)
        return minimum_error, minimum_iteration, step_size_and_Precision_with_minimum_error
