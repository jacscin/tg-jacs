import matplotlib.pyplot as plt
import numpy as np
import itertools
import time
import random
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.metrics import make_scorer

def mse(Ay, Py):
    # Lower is better
    return np.mean((Ay - Py) ** 2)

def mape(Ay, Py):
    # Lower is better
    localAy = np.array(Ay)
    localAy[np.where(Ay == 0)] = Py[np.where(Ay == 0)] / (1 - Py[np.where(Ay == 0)])
    return (100 * np.mean(np.abs((localAy - Py) / localAy)))

def nmse(Ay, Py):
    # Lower is better
    return (np.sum((Ay - Py) ** 2) / np.sum(np.diff(Py) ** 2))

def pocid(Ay, Py):
    # Higher is better
    return (100 * np.mean(list(map(int, np.diff(Ay)*np.diff(Py) > 0))))

def arv(Ay, Py):
    # Lower is better
    return (np.sum((Py - Ay) ** 2) / np.sum((Py - np.mean(Ay)) ** 2))

def fitness(Ay, Py):
    return (pocid(Ay, Py) / (1 + mse(Ay, Py) + mape(Ay, Py) + nmse(Ay, Py) + arv(Ay, Py)))

# Load data
bbas3 = np.loadtxt('BBAS3_COTAHIST_A2010-2019.txt')
bbdc4 = np.loadtxt('BBDC4_COTAHIST_A2010-2019.txt')
itub4 = np.loadtxt('ITUB4_COTAHIST_A2010-2019.txt')
sanb11 = np.loadtxt('SANB11_COTAHIST_A2010-2019.txt')

# Choose parameters to test
parameters = {'kernel': ['rbf'],
              'C': [1, 100, 1000, 10000],
              'epsilon': [0.1, 0.01, 0.001],
              'gamma': [0.1, 0.01, 0.001, 0.0001]}

# Make scorer to training set
fit_score = make_scorer(fitness)

# Make alternate lags to test
target = 10
perms = [i for i in range(target)]
lags = []
for L in range(1, target+1):
    for subset in itertools.combinations(perms, L):
        lags.append(np.array(subset))

lags = np.array(lags, dtype=object)

# Choose data to work
for data in [bbas3, bbdc4, itub4, sanb11]:
    # Show the dataset tested now
    if (data == bbas3).all():
        print("BBAS3")
    if (data == bbdc4).all():
        print("BBDC4")
    if (data == itub4).all():
        print("ITUB4")
    if (data == sanb11).all():
        print("SANB11")

    # Define division point in data between train and test set
    divPoint = int((data.size - target)*0.7)

    # Normalize data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Variable to store best model between parameters and alternate lag options
    bestLag = {
        'C': 0,
        'epsilon': 0,
        'gamma': 0,
        'lag': [],
        'Fitness': 0
    }

    # Starting counting training time
    start_time = time.time()
    for lag in lags:
        # Prepare data
        X = np.array([data[lag+i] for i in range(data.size - target - 1)])
        Y = np.array([data[i+target] for i in range(1, data.size - target)])
        # Divide data into train and test
        Xf = np.array(X[:divPoint])
        Yf = np.array(Y[:divPoint])
        Xt = np.array(X[divPoint:])
        Yt = np.array(Y[divPoint:])

        # Train model
        grdSrch = GridSearchCV(SVR(), parameters, n_jobs = -1, scoring=fit_score, cv = TimeSeriesSplit())
        grdSrch.fit(Xf, Yf)

        # Store best
        Yp = grdSrch.predict(Xt)
        fit = fitness(Yt, Yp)
        if(fit > bestLag['Fitness']):
            bestLag['C'] = grdSrch.cv_results_['params'][grdSrch.best_index_]['C']
            bestLag['epsilon'] = grdSrch.cv_results_['params'][grdSrch.best_index_]['epsilon']
            bestLag['gamma'] = grdSrch.cv_results_['params'][grdSrch.best_index_]['gamma']
            bestLag['lag'] = lag
            bestLag['Fitness'] = fit

    print("--- %s seconds ---" % (time.time() - start_time))
    print(bestLag)
