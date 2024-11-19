from sklearn import preprocessing
import pandas as pd


def load_dataset(pass_of_dataset, numberofprocessing):
    def min_max_scaller(data):
        min_data = data.min()
        max_data = data.max()
        data = (data-min_data)/(max_data-min_data)
        return data.to_numpy()

    def standrlization(data):
        return (data-data.mean())/(data.std())
    df = pd.read_csv(pass_of_dataset)
    examples, features = df.shape
    if (numberofprocessing == 0):
        return (df.iloc[:, 0:-1].to_numpy(), df.iloc[:, -1].to_numpy())
    elif (numberofprocessing == 1):
        return min_max_scaller(df.iloc[:, 0:-1]), min_max_scaller(df.iloc[:, -1])
    else:
        return standrlization(df.iloc[:, 0:-1]), standrlization(df.iloc[:, -1])
