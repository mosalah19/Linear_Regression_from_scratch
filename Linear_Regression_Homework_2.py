import argparse
import numpy as np
import matplotlib as plt
from predict_and_optimized_hyperparameter import *
from load_and_processing import *
from normal_equation import *

from LR_using_Gradient_Descent import *

parser = argparse.ArgumentParser('linear regression homework 2')
parser.add_argument('--dataset', type=str,
                    default=r'D:\Data Science\course_ML_mostafa_saad\03-linear regression\practice\dataset_200x4_regression.csv', help='Path to the CSV dataset file')
parser.add_argument('--preprocessing', type=str, default=1,
                    help='0 for no preprocessing,'
                    '1 for min /max scalling,'
                    '2 for standlizing,'
                    )
parser.add_argument('--choice', type=str, default=3,
                    help='0 for linear verification,'
                    '1 for trannin with all features,'
                    '2 for tranning with the best features,'
                    '3 for normal equation,'
                    '4 for sikit,')
parser.add_argument('--step_size', type=float, default=0.01,
                    help='Learning rate (default 0.01)')
parser.add_argument('--precision', type=float, default=0.0001,
                    help='Requested precision (default: 0.0001)')
parser.add_argument('--max_iter', type=int, default=10000,
                    help='number of epochs to train (default: 1000) ')

args = parser.parse_args()
if args.choice == 0:
    X = np.array([0, 0.2, 0.4, 0.8, 1.0])
    t = X+5
    X = X.reshape((-1, 1))
else:
    X, T = load_dataset(args.dataset, args.preprocessing)
X = np.hstack([np.ones((X.shape[0], 1)), X])
if args.choice == 0:
    optimal_weights, visited_points, cost_history, cost, iterative = linear_regression_using_gradient_descent(
        X, t, 0.1, 0.00001, 1000)


def visulization1(X, t, optimal_weights):
    predict = np.dot(X, optimal_weights)
    plt.scatter(X[:, 1:], t)
    plt.plot(X, predict)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


if args.choice == 1:
    optimal_weights, visited_points, cost_history, cost, iterative = linear_regression_using_gradient_descent(
        X, T, args.step_size, args.precision, args.max_iter)
    visulization(cost_history, iterative)
    print(cost, iterative)

if args.choice == 2:
    df = pd.read_csv(args.dataset)
    Investigation(df)

    # from visulization feature 1 => model only using this feature
    optimal_weights, visited_points, cost_history, cost, iterative = linear_regression_using_gradient_descent(
        X[:, 0:2], T, args.step_size, args.precision, args.max_iter)

    prediction(X[:, 0:2], T, optimal_weights)
    visulization1(X[:, 0:2], T, optimal_weights)
if args.choice == 3:
    optimals_waights = Normal_equation(X, T)

elif args.choice == 4:
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error

    # LinearRegression from sklearn uses OLS (normal equations)
    # We can remove our added 1 column and use fit_intercept = True
    X_ = X[:, 1:]
    model = linear_model.LinearRegression(fit_intercept=True)

    # Train the model using the training sets
    model.fit(X_, t)
    optimal_weights = np.array([model.intercept_, *model.coef_])
    print(f'Scikit parameters: {optimal_weights}')

    # Make predictions using the testing set
    pred_t = model.predict(X_)
    # divide by 2 to MATCH our cost function
    error = mean_squared_error(t, pred_t) / 2
    print(f'Error: {error}')
