import numpy as np

from sklearn.model_selection import learning_curve,validation_curve,train_test_split

import sklearn.metrics as metrics

def classifier_lc(X,y,estimator,model_name): #learning curve

    train_sizes = np.linspace(1,X.shape[0]*0.7, 5,dtype=int)
    train_sizes, train_scores, validation_scores = learning_curve(
                                    estimator = estimator(),
                                        X = X,y = y, train_sizes = train_sizes, cv = 4,scoring = 'accuracy')
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)

    return {
    "curves":
     [
        {"x":list(train_sizes),"y":list(train_scores_mean),"label":"Training accuracy"},
        {"x":list(train_sizes),"y":list(validation_scores_mean),"lablel":"Validation accuracy"}
    ],
    "info":  {"xlabel":"Training set size" ,"ylabel":"accuracy","title":"Learning curves for a "+model_name}
    }



def classifier_vc(X,y,estimator,model_name): #validation curve


    param_name="max_depth"
    if(model_name ==""):
        param_name="max_iter"

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
        {"x":list(param_range),"y":list(train_scores_mean),"label":"Training score"},
        {"x":list(param_range),"y":list(test_scores_mean),"lablel":"Cross-validation score"}
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
        {"x":list(fpr),"y":list(tpr),"lablel":label}
    ],
    "info": {"xlabel":"True Positive Rate" ,"ylabel":"True Positive Rate","title":model_name+" ROC"}
    }