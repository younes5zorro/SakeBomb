#  **************RFR : Random Forest Regressor ************    /
#___________________________________________________________/



__author__ = "EA"

import pandas as  pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#--Metrics To Evaluate Machine Learning Algorithm:

# from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error,r2_score

# from feature_selector import FeatureSelector

import json
from Models import utils


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
    clf = DecisionTreeRegressor()
    clf.fit(X_train,y_train)
    return clf


def TrainingFine_tunning(X_train, y_train,parameters):
    if(parameters != None):
        clf = DecisionTreeRegressor()
        clf.set_params(**parameters)
        clf.fit(X_train,y_train)
        return clf



#***************************Auto-tuning : choix automatique des macro-paramètres de DecisionTreeRegressor
#______________________________________________________________________________________|

def autotuning(X_train, y_train):


    parameters={'min_samples_split' : range(5,500,20),'max_depth': range(1,20,2)}

    clf_RFR= DecisionTreeRegressor()
    clf=GridSearchCV(clf_RFR,parameters)
    clf.fit(X_train,y_train)
   # print(clf.best_estimator_)
    return clf

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

    data["mean_squared_error_Test"] = mean_squared_error(y_test,predict_test)

    #log_loss["roc_auc_score_Test"] = log_loss(y_test,predict)
    data["variance_score_Test"] = r2_score(y_test,predict_test)

    if data["variance_score_Test"] >= 0.8:
            data["Etat_Test"] = "Excellent"
    elif data["variance_score_Test"] >= 0.6:
            data["Etat_Test"] = "Moyen"
    else :
            data["Etat_Test"] = "Mauvais"
 
    data["mean_squared_error_Val"] = mean_squared_error(y_val,predict_val)
    data["variance_score_Val"] = r2_score(y_val,predict_val)

    if data["variance_score_Val"] >= 0.8:
            data["Etat_Val"] = "Excellent"
    elif data["variance_score_Val"] >= 0.6:
            data["Etat_Val"] = "Moyen"
    else :
            data["Etat_Val"] = "Mauvais" 
    return data

# #--Variable Selection

# def  feature_selector(dataFrame,train_labels):

#     fs = FeatureSelector(data = dataFrame, labels = train_labels)
#     fs.identify_missing(missing_threshold = 0.6)

#     '''This method finds pairs of collinear features based on the Pearson correlation coefficient.
#     For each pair above the specified threshold (in terms of absolute value),
#     it identifies one of the variables to be removed. '''

#     fs.identify_collinear(correlation_threshold = 0.98)
#     fs.identify_zero_importance(task = 'classification',
#                             eval_metric = 'auc',
#                             n_iterations = 10,
#                              early_stopping = True)

#     # list of zero importance features
#     zero_importance_features = fs.ops['zero_importance']

#     #Once we have identified the features to remove: feature with missing values, feauture with low importance
#     train_no_missing_zero = fs.remove(methods = ['missing', 'zero_importance'])

#     all_to_remove = fs.check_removal()

#     return train_no_missing_zero
