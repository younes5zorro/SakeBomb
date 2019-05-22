#      ************Decision Tree*****************            /
#___________________________________________________________/
__author__ = "EA"
# -*- coding: utf-8 -*-

# import sys
# sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from SciPy packages
from statsmodels.tsa.stattools import adfuller # adfuller test
from statsmodels.graphics.tsaplots import plot_acf # autocorellation plot
from statsmodels.graphics.tsaplots import plot_pacf # partial autocorellation plot

# math function
from math import sqrt

# evaluation metrics
from sklearn.metrics import mean_squared_error

# statistics models
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# libraries to filter warnings in some algorithms
import warnings
warnings.filterwarnings("ignore")

# from feature_selector import FeatureSelector

# from sklearn.model_selection import cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split

# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import log_loss
# from sklearn.model_selection import GridSearchCV

# #Tree visualization library
# # from sklearn.externals.six import StringIO
# # from IPython.display import Image
# from sklearn.tree import export_graphviz
# import pydotplus
# import json
# from Models import utils



# split a univariate dataset into train/test sets
def split_dataset(data, n_test):
    # split into standard weeks
    train, test = data[0:-n_test], data[-n_test:]
    # restructure into windows of weekly data
    train = np.array(np.split(train, len(train)/7))
    test = np.array(np.split(test, len(test)/7))
    return train, test

# convert windows of weekly multivariate data into a series of closing price
def to_series(data):
    # extract just the price of XRP from each week
    series = [week[:, 0] for week in data]
    # flatten into a single series
    series = np.array(series).flatten()
    return series

# Arima forecast for weekly prediction
def arima_forecast(history, arima_order):
    # convert history into a univariate series
    series = to_series(history)
    # define the model
    model = ARIMA(series, order=arima_order)
    # fit the model
    model_fit = model.fit(disp=False)
    # make forecast
    yhat = model_fit.predict(len(series), len(series)+6)
    return yhat


# Sarima forecast for weekly prediction
def Sarima_forecast(history, config):
    order, sorder, trend = config
    # convert history into a univariate series
    series = to_series(history)
    # define model
    model = SARIMAX(series, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(series), len(series)+6)
    return yhat

# evaluate one or more weekly forecasts against expected values
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

def evaluate_model(model_func, train, test, *args):
    #history of weekly data
    history = [x for x in train]
    #walk forward validation
    predictions = list()
    for i in range(len(test)):
    #weekly prediction
        y_hat_seq = model_func(history, *args)
    #store the preditions
        predictions.append(y_hat_seq)
    #update history data
        history.append(test[i,:])
    predictions = np.array(predictions)
    # evaluate predictions days for each week
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores, predictions

# Return score on ARIMA family model, for model assessment
def evaluate_arima_family_scores(func, name, train, test, order, debug = False):
    score = None
    # show all warnings and fail on exception if debugging
    if debug:
        score, scores, predictions = evaluate_model(func, train, test, order) #evaluate particular model with walk-forward validation
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                score, scores, predictions = evaluate_model(func, train, test, order)
        except:
            score = None 
    # check for an interesting result
    if score is not None: # won't print model configurations that returned nothing
        print(name + '%s RMSE=%.3f' % (order,score))
    return (order, score)

# grid search configs for ARIMA model
def grid_search_arima_family(func, name, train, test, orders_list, parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(evaluate_arima_family_scores)(func, name, train, test, order) for order in orders_list)
        scores = executor(tasks)
    else:
        scores = [evaluate_arima_family_scores(func, name, train, test, order) for order in orders_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores, name

 # create a set of sarima configs to try
def arima_orders(p_values, d_values, q_values):
    orders = list()
    # create config instances
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                orders.append(order)
    return orders
   
#***********Data spliting


def splitData(dataFrame, trainTestValidation):

    X=dataFrame.iloc[:,:-1].values
    y=dataFrame.iloc[:,-1].values
    #***************Split data into train,test

    if(trainTestValidation != None):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=trainTestValidation['test_size'], random_state=1)

    #***************split train again into validation and train
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=trainTestValidation['validation_size'], random_state=1)

        return X_train, y_train, X_val, y_val, X_test, y_test



def TrainingDefaultParameters(X_train, y_train):

    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    return clf


def TrainingFine_tunning(X_train, y_train,parameters):
    if(parameters != None):
        clf = DecisionTreeClassifier()
        clf.set_params(**parameters)
        clf.fit(X_train,y_train)
        return clf



#***************************Auto-tuning : choix automatique des macro-paramètres de Decision Tree
#________________________________________________________________________________________________|


def autotuning(X_train, y_train):


    parameters={'min_samples_split' : range(5,500,20),'max_depth': range(1,20,2)}
    clf_tree=DecisionTreeClassifier()
    clf=GridSearchCV(clf_tree,parameters)
    clf.fit(X_train,y_train)
    #print(clf.best_estimator_)
    trained_model=clf.best_estimator_.fit(X_train,y_train)
    return trained_model


#**************************Make prediction
#__________________________________________|
def testSetPrediction(X_test,X_val,clf):

    predict_test=clf.predict(X_test)
    predict_val=clf.predict(X_val)
    return predict_test,predict_val



#**************************Evaluation
#____________________________________|
def scoring(y_test,predict_test,y_val,predict_val,clf):

    data={}

#     data["accuracy_score_Test"] = accuracy_score(y_test,predict_test)
#     data["roc_auc_score_Test"] = log_loss(y_test,predict_test)
    data["Model"]  =""    
    data["Status"]  ="Train/validation"
    data["Accuracy Trainning"] = round(accuracy_score(y_test,predict_test),2)
    if data["Accuracy Trainning"] >= 0.8:
            data["Etat Trainning"] = "Excellent"
    elif data["Accuracy Trainning"] >= 0.6:
            data["Etat Trainning"] = "Moyen"
    else :
            data["Etat Trainning"] = "Mauvais"  

    data["Accuracy Validation"] =  round(accuracy_score(y_val,predict_val),2)

    if data["Accuracy Validation"] >= 0.8:
            data["Etat Validation"] = "Excellent"
    elif data["Accuracy Validation"] >= 0.6:
            data["Etat Validation"] = "Moyen"
    else :
            data["Etat Validation"] = "Mauvais"    
    #print (data)

    return data

#Variable Selection
def  feature_selector(dataFrame,train_labels):

    fs = FeatureSelector(data = dataFrame, labels = train_labels)
    fs.identify_missing(missing_threshold = 0.6)

    '''This method finds pairs of collinear features based on the Pearson correlation coefficient.
    For each pair above the specified threshold (in terms of absolute value),
    it identifies one of the variables to be removed. '''

    fs.identify_collinear(correlation_threshold = 0.98)
    fs.identify_zero_importance(task = 'regression',
                            eval_metric = 'auc',
                            n_iterations = 10,
                             early_stopping = True)

    # list of zero importance features
    zero_importance_features = fs.ops['zero_importance']

    #we have identified the features to remove: feature with missing values, feauture with low importance
    train_no_missing_zero = fs.remove(methods = ['missing', 'zero_importance'])

    all_to_remove = fs.check_removal()

    return train_no_missing_zero



# def visualization(dtree):
#     dot_data = StringIO()
#     export_graphviz(dtree, out_file=dot_data,
#                     filled=True, rounded=True,
#                     special_characters=True)
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#     image=Image(graph.create_png())
#     utils.ensure_dir("output/visualisation")
#     Image(graph.write_png('output/visualisation/Tree_visu.png'))
#     return image


