import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


def forest():
    path = 'csv/M_all.csv'
    df = pd.read_csv(path)

    x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    feat_labels = df.columns[1:]
    forest = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    forest.fit(x_train, y_train)
    data = pd.DataFrame(columns=['name'])
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(x_train.shape[1]):
        data = data.append({'name': feat_labels[indices[f]]}, ignore_index=True)
    data.to_csv('data.csv', index=False)


def process():  #
    path = 'csv/'
    df = pd.read_excel(path + 'Molecular_all.xlsx')
    df.to_csv(path + 'Molecular_all.csv', index=False)
    col = df.columns
    for n in col:
        if df[n].std() == 0:
            df.pop(n)


def correlation():
    path = 'csv/Molecular_all.csv'
    df = pd.read_csv(path)
    col = df.columns
    for n in col:
        pass
    a = 1


if __name__ == '__main__':
    forest()
