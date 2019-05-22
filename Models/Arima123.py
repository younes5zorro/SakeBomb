
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

from Models import utils
from math import sqrt

#***********Data spliting

def splitData(X, trainTestValidation):

    X = X.values
    size = int(len(X) * trainTestValidation['test_size'])
    train, test = X[0:size], X[size:len(X)]

    return train, test


# def TrainingDefaultParameters(X_train, order):

#     clf = ARIMA()
#     clf.fit(X_train.tolist())
#     return clf


def arima_orders(p_values, d_values, q_values):
    orders = list()
    # create config instances
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                orders.append(order)
    return orders

def grid_search_arima_family(func, name, train, test, orders_list, parallel=True):
    scores = None
    
    scores = [evaluate_arima_family_scores(func, name, train, test, order) for order in orders_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores, name

# Return score on ARIMA family model, for model assessment
def evaluate_arima_family_scores(func, name, train, test, order):
    score = None
    score, scores, predictions = evaluate_model(func, train, test, order) #evaluate particular model with walk-forward validation

    if score is not None: # won't print model configurations that returned nothing
        print(name + '%s RMSE=%.3f' % (order,score))
    return (order, score)


def evaluate_model( model_func, train, test, *args):

    history = [x for x in train]

    predictions = list()
    for i in range(len(test)):

        y_hat_seq = model_func(history, *args)

        predictions.append(y_hat_seq)

        history.append(test[i,:])

    predictions = np.array(predictions)

    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores, predictions

def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores

def autotuning(X_train, y_train):

    p_values = range(1, 2)
    d_values = range(0, 2)
    q_values = range(1, 2)

    orders_arima_list = arima_orders(p_values,d_values,q_values)

    scores, name = grid_search_arima_family(func, name, train, test, arg, parallel=False)

    return trained_model

series = pd.Series.from_csv('../uploads/births.csv', header=0)
# prepare data
X = series.values
X = X.astype('float32')
# fit model
model = ARIMA(X, order=(1,1,1))
model_fit = model.fit()
# save model
model_fit.save('../saved/model.pkl')
# load model
loaded = ARIMAResults.load('../saved/model.pkl')