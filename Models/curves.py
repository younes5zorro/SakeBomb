import numpy as np

from sklearn.model_selection import learning_curve,validation_curve,train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

from imgurpython import ImgurClient
import os
from io import BytesIO
import pandas as pd
import requests

def get_estimator(model_name):
    model  = {}
    if(model_name == "Arbre de decision") :
        from sklearn.tree import DecisionTreeClassifier as DT
        model=DT

    if(model_name == "Random Forest") :
        from sklearn.ensemble import RandomForestClassifier as RFC
        model=RFC

    if(model_name == "Random Forest Regression") :
        from sklearn.tree import DecisionTreeRegressor as RFR
        model=RFR

    if(model_name == "Kmeans") :
        from sklearn.cluster import KMeans as KM
        model=KM

    if(model_name =="Régression Liniaire") :
        from sklearn.linear_model import LinearRegression as LinReg
        model=LinReg

    if(model_name == "SVM") :
        from sklearn.svm import SVC as svm
        model=svm

    if(model_name == "Decision tree regression") :
        from sklearn.tree import DecisionTreeRegressor as DTReg
        model=DTReg

    if(model_name == "régression logistique") :
        from sklearn.linear_model import LogisticRegression as LogReg
        model=LogReg

    if(model_name == "xgboost classification") :
        from xgboost import XGBClassifier as XgbC
        model=XgbC
    
    if(model_name == "xgboost regression") :
        from xgboost import XGBRegressor as XgbReg
        model=XgbReg

    return model

def get_graphs(json_data):
        

        dataFrame,train_labels = get_dataFram(json_data["link"],json_data["input"],json_data["output"])

        trainTestValidation=json_data['trainTestValidation']
        model_name  = json_data["model_name"]
        model_type  = json_data["model_type"]
        
        model = get_estimator(model_type)

        if(json_data['Feautue_Selection'] == True ):
                 dataFrame=model.feature_selector(dataFrame, train_labels)

        X_train, y_train, X_val, y_val, X_test, y_test=model.splitData(dataFrame, trainTestValidation)

        
        if (json_data['Operation'] == "Default_Parameters"):
                clf= model.TrainingDefaultParameters(X_train, y_train)

        #Training of the model using parameters entered by the user
        elif(json_data['Operation'] == "Fine_tuning"):
                parameters=json_data['parameters']
                clf= model.TrainingFine_tunning(X_train, y_train,parameters)

        #Training of the model using autotuning
        elif(json_data['Operation'] == "autotuning") :
                clf= model.autotuning(X_train, y_train)

        predict_test,predict_val=model.testSetPrediction(X_test,X_val, clf)
        score=model.scoring(y_test,predict_test,y_val,predict_val,clf)

        filename = str(score["Accuracy Trainning"])+"~~"+str(score["Accuracy Validation"])+"~~"+model_name+".pkl"
        # filename = model_name+"_"+str(randint(0, 3000))+".pkl"

        score["Model"] = model_name

        if model_type in ["Régression Liniaire"] :
                score["x"]=X_test.reshape(1,-1)[0].tolist()
                # score["x"]=X_test.reshape(1,X_test.shape[0])[0].tolist()
                score["y"]=y_test.tolist()

        joblib.dump(clf, MODELS_FOLDER / filename)
        # pickle.dump(clf, open(MODELS_FOLDER / filename , 'wb'))

        # Step 5: Return the response as JSON
        return score

def classifier_lc(X,y,estimator,model_name): #learning curve

    train_sizes = np.linspace(1,X.shape[0]*0.7, 5,dtype=int)
    train_sizes, train_scores, validation_scores = learning_curve(
                                    estimator = estimator(),
                                        X = X,y = y, train_sizes = train_sizes, cv = 4,scoring = 'accuracy')
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)

    result = {"curves":[{"x":train_sizes.tolist(),"y":train_scores_mean.tolist(),"label":"Training accuracy"},
        {"x":train_sizes.tolist(),"y":validation_scores_mean.tolist(),"lablel":"Validation accuracy"}],
        "info":  {"xlabel":"Training set size" ,"ylabel":"accuracy","title":"Learning curves for a "+model_name}
    }

    return result

def classifier_vc(X,y,estimator,model_name,param_name): #validation curve

    param_range = np.linspace(1,300, 20,dtype=int)
    train_scores, test_scores = validation_curve(
        estimator(), X, y,  param_name=param_name, param_range=param_range,
        cv=5, scoring="accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    return {
    "curves":
     [
        {"x":param_range.tolist(),"y":train_scores_mean.tolist(),"label":"Training score"},
        {"x":param_range.tolist(),"y":test_scores_mean.tolist(),"lablel":"Cross-validation score"}
    ],
    "info":  {"xlabel":"depth" ,"ylabel":"accuracy","title":"Validation Curves for a "+model_name}
    }

def classifier_roc(X,y,estimator,model_name): #learning curve
    model=estimator()
    model.fit(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    label = 'AUC = %0.2f' % roc_auc
    return {
    "curves":
     [
        {"x":[0, 1],"y":[0, 1],"label":""},
        {"x":fpr.tolist(),"y":tpr.tolist(),"lablel":label}
    ],
    "info": {"xlabel":"True Positive Rate" ,"ylabel":"True Positive Rate","title":model_name+" ROC"}
    }

def regressor_vc(X,y,estimator,model_name,param_name): #learning curve
	param_range = np.linspace(1,50, 10,dtype=int)
	print(param_range)
	train_scores, test_scores = validation_curve(
	    estimator(), X, y,  param_name=param_name, param_range=param_range,
	    cv=5, scoring="neg_mean_squared_error")
	train_scores_mean = -np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = -np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	return {
    "curves":
     [
      {"x":param_range.tolist(),"y":train_scores_mean.tolist(),"label":"Training score"},
      {"x":param_range.tolist(),"y":test_scores_mean.tolist(),"lablel":"Cross-validation score"},
    ],
    "info": {"xlabel":"depth" ,"ylabel":"accuracy","title":"Validation Curve for a "+model_name}
    }

def regressor_lc(X,y,estimator,model_name): #learning curve
	train_sizes = np.linspace(1,X.shape[0]*0.8, 50,dtype=int)
	train_sizes, train_scores, validation_scores = learning_curve(
	                                 estimator = estimator(),
	                                      X = X,
	     y = y, train_sizes = train_sizes, cv = 5,
	                scoring = 'neg_mean_squared_error')
	train_scores_mean = -train_scores.mean(axis = 1)
	validation_scores_mean = -validation_scores.mean(axis = 1)
	return {"curves":
     [
       {"x":train_sizes.tolist(),"y":train_scores_mean.tolist(),"label":"Training error"},
       {"x":train_sizes.tolist(),"y":validation_scores_mean.tolist(),"lablel":"Validation error"},
    ],
    "info": {"xlabel":"Training set size" ,"ylabel":"Error","title":"Learning curves for a "+model_name}
    }

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          path=""
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    client_id = '799656b0ea972d6'
    client_secret = '953bb107494227f9d350cc918ea33c6bd8c841b5'

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    print("claasses",classes)

    classes = classes[unique_labels(y_true, y_pred)]
    print("claasses",classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    
    client = ImgurClient(client_id, client_secret)
    # plt.show()
    plt.savefig(path)
    c= client.upload_from_path(path, config=None, anon=True)
    os.remove(path)
    return c["link"]
    # return [ {"matrice":cm.tolist() , "classes":classes} ]

def classifier_cm(X,y,estimator,model_name): #learning curve

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    y_pred = estimator().fit(X_train, y_train).predict(X_test)
    class_names=np.array(list(y))
    link = plot_confusion_matrix(y_test, y_pred, classes=class_names,
                        title='Confusion matrix',path="tttttttttest.png")

    return link
    # plt.savefig("CM_DecisTreeClass.png")
    # plt.show()