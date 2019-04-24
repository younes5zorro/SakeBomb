#      ************Decision Tree*****************            /
#___________________________________________________________/
__author__ = "EA"
# -*- coding: utf-8 -*-

# import sys
# sys.path.append("..")
from feature_selector import FeatureSelector

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV

#Tree visualization library
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# from sklearn.tree import export_graphviz
# import pydotplus
import json
from Models import utils



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
