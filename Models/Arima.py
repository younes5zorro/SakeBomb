
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

from Models import utils
from math import sqrt

#***********Data spliting

def splitData(X, test_size):

    X = X.values
    size = int(len(X) * test_size)
    train, test = X[0:size], X[size:]

    return train, test
 
def arima_orders(p_values, d_values, q_values):
    orders = list()
    # create config instances
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                orders.append(order)
    return orders

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X,test_size, arima_order):
	# prepare training dataset
	train, test =splitData(X, test_size)

	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, test_size,p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for order in arima_orders(p_values, d_values, q_values):
            # try:
                mse = evaluate_arima_model(dataset,test_size, order)
                if mse < best_score:
                    best_score, best_cfg = mse, order
                print('ARIMA%s MSE=%.3f' % (order,mse))
            # except:
            #     print("apqss")
            #     continue
                
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))