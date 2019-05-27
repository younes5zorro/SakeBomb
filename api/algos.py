from flask import Blueprint, request, jsonify
import pathlib
import os
from random import randint
import pandas as  pd
import petl as etl 
import shutil
 
from sklearn.externals import joblib
from io import BytesIO
import requests

import pandas as pd
import numpy as np

import Models.curves as curves

advance_alogs = Blueprint('advance_alogs', __name__)

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = PACKAGE_ROOT / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

MODELS_FOLDER = PACKAGE_ROOT / 'saved'
MODELS_FOLDER.mkdir(exist_ok=True)
  

def get_from_excel(link,sheet):

        file = requests.get(link).content
        
        df =pd.read_excel(BytesIO(file),sheet_name =sheet,parse_dates=True)
        df = df.replace(np.nan, '', regex=True)
        tab = etl.fromdataframe(df)

        return tab

def get_from_csv(link):

        file = requests.get(link).content
        
        df =pd.read_csv(BytesIO(file),parse_dates=True)

        tab = etl.fromdataframe(df) 

        return tab

def get_df_from_excel(link,sheet):

        file = requests.get(link).content
        
        df =pd.read_excel(BytesIO(file),sheet_name =sheet)

        return df

def get_df_from_csv(link):

        file = requests.get(link).content
        
        df =pd.read_csv(BytesIO(file))

        return df

@advance_alogs.route('/algos', methods=['GET'])
def algos():
    if request.method == 'GET':
        return jsonify({'algos_version': '0.0.0',
                        'algos_api_version': '1.1.1'})

@advance_alogs.route('/v1/delete_model', methods=['POST'])
def delete_model():
        if request.method == 'POST':
                json_data = request.get_json()

                model_name = json_data["model_name"]        
                fil = [x.unlink() for x in MODELS_FOLDER.glob('*') if x.is_file() and x.name.split('~~')[2]==model_name+'.pkl']
                
                return jsonify({"message":model_name + " removed"})

@advance_alogs.route('/v1/list_models', methods=['GET'])
def list_models():
       
        files = [x.name.replace(".pkl","").split('~~')[2] for x in MODELS_FOLDER.glob('*') if x.is_file()]

        return jsonify(files)

@advance_alogs.route('/v1/clear', methods=['GET'])
def clear():
        shutil.rmtree(MODELS_FOLDER)
        MODELS_FOLDER.mkdir(exist_ok=True)
        return jsonify({"message":" removed"})

def get_dataFram(json_data):

        df = {}
        if  json_data['type'] =="excel":
                df = get_df_from_excel(json_data['link'],json_data['sheet'])
        elif  json_data['type'] =="csv":
                df = get_df_from_csv(json_data['link'])

        input_field  = json_data["input"]
        input_field.append(json_data["output"])

        df = df[input_field]

        train_labels=df.iloc[:,-1]

        return df,train_labels

def get_model(model_name):
    model  = {}
    if(model_name == "Arbre de decision") :
        from Models import DecisionTree as DT
        model=DT

    if(model_name == "Random Forest") :
        from Models import randomForestClassifier as RFC
        model=RFC

    if(model_name == "Random Forest Regression") :
        from Models import randomForestRegressor as RFR
        model=RFR

    if(model_name == "Kmeans") :
        from Models import KMeans as KM
        model=KM

    if(model_name =="Régression Liniaire") :
        from Models import LinearRegression as LinReg
        model=LinReg

    if(model_name == "SVM") :
        from Models import SVM as svm
        model=svm

    if(model_name == "Decision tree regression") :
        from Models import DecisionTreeRegressor as DTReg
        model=DTReg

    if(model_name == "régression logistique") :
        from Models import LogisticRegression as LogReg
        model=LogReg

    if(model_name == "xgboost classification") :
        from Models import XGBClassifier as XgbC
        model=XgbC
    
    if(model_name == "xgboost regression") :
        from Models import XGBRegressorRegressor as XgbReg
        model=XgbReg

    return model

def get_score(json_data):
        
        result ={}
        dataFrame,train_labels = get_dataFram(json_data)

        trainTestValidation=json_data['trainTestValidation']
        model_name  = json_data["model_name"]
        model_type  = json_data["model_type"]
        
        model = get_model(model_type)

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
        

        result['score'] = score
        if model_type in ["Régression Liniaire"] :
                datax=X_test.reshape(1,-1)[0].tolist()
                # score["x"]=X_test.reshape(1,X_test.shape[0])[0].tolist()
                datay=y_test.tolist()

                result['data']  = [list(a) for a in zip(datax,datay)]

        joblib.dump(clf, MODELS_FOLDER / filename)
        # pickle.dump(clf, open(MODELS_FOLDER / filename , 'wb'))

        # Step 5: Return the response as JSON
        return result

@advance_alogs.route('/v1/train', methods=['POST'])
def train_model():
    if request.method == 'POST':
        json_data = request.get_json()
        
        score = get_score(json_data)
        return jsonify(score)
        
@advance_alogs.route('/v1/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        model_name  = json_data["model_name"]
        link  = json_data["link"]

        files = [x.name for x in MODELS_FOLDER.glob('*') if x.is_file() and x.name.split('~~')[2]==model_name+'.pkl']

        loaded_model = joblib.load(MODELS_FOLDER / files[0])

        df,train_labels = get_dataFram(json_data["link"])
        pred_cols = list(df.columns.values)[:-1]
        # pred_cols = list(pr.columns.values)

        # apply the whole pipeline to data
        pred = list(pd.Series(loaded_model.predict(df[pred_cols].values)))

        di ={0:"dissatisfied",1:"satisfied"}

        pred =  list(map(di.get, pred))

        df[list(df.columns)[-1]] = pred

        return jsonify(df.to_dict('records'))

@advance_alogs.route('/v1/compare', methods=['POST'])
def compare():
    if request.method == 'POST':
        json_data = request.get_json()
        models  = json_data["models"]

        result = []
        for model_name in models:
                doc = {}
                doc["model"] = model_name
                files = [x.name for x in MODELS_FOLDER.glob('*') if x.is_file() if  len(x.name.split('~~'))> 1 and x.name.split('~~')[2]==model_name+'.pkl']
                file = files[0]
                split = file.split('~~')
                doc["Accuracy Trainning"] = split[0]
                doc["Accuracy Validation"] = split[1]

                result.append(doc)
                

        return jsonify(result)

@advance_alogs.route('/v1/selection', methods=['POST'])
def features_selection():
        if request.method == 'POST':
                from sklearn.linear_model import Lasso
                from sklearn.feature_selection import SelectFromModel

                json_data = request.get_json()

                result ={}
                X_train, y_train= get_ForSelection(json_data)

                sel_ = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
                sel_.fit(X_train, y_train)

                selected_feat = X_train.columns[(sel_.get_support())]

                print('total features: {}'.format((X_train.shape[1])))
                print('selected features: {}'.format(len(selected_feat)))
                print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_ == 0)))

                result["labels"] = list(X_train.columns)
                result["values"] = list(np.around(np.abs(sel_.estimator_.coef_)*100, decimals=2))

                return jsonify(result)

@advance_alogs.route('/v1/kmean', methods=['POST'])
def kmeandata():
        if request.method == 'POST':

                json_data = request.get_json()
                # link = json_data["link"]
                header = json_data["header"]

                # file = requests.get(link).content
        
                df,l =get_dataFram(json_data)

                df = df[header]

                result = df.values.tolist()

                return jsonify(result)

def get_ForSelection(json_data):

        df ,l =get_dataFram(json_data)

        X_train = df.iloc[:,:-1]  #independent columns
        y_train = df.iloc[:,-1] 

        return X_train,y_train

@advance_alogs.route('/v1/l_curve', methods=['POST'])
def l_curve():
    if request.method == 'POST':

        result ={}    
        json_data = request.get_json()
        
        dataFrame,train_labels = get_dataFram(json_data)

        X = dataFrame.iloc[:,:-1] 
        Y = train_labels

        model_name  = json_data["model_name"]
        model_type  = json_data["model_type"]

        estimator  = curves.get_estimator(model_type)

        if model_type in ["Arbre de decision","Random Forest","SVM","régression logistique","xgboost classification"]:
                result = curves.classifier_lc(X,Y,estimator,model_name)
        else:
                result = curves.regressor_lc(X,Y,estimator,model_name)

        return jsonify(result)

@advance_alogs.route('/v1/v_curve', methods=['POST'])
def v_curve():
    if request.method == 'POST':

        result ={}    
        json_data = request.get_json()
        
        dataFrame,train_labels = get_dataFram(json_data)

        X = dataFrame.iloc[:,:-1] 
        Y = train_labels

        model_name  = json_data["model_name"]
        model_type  = json_data["model_type"]

        estimator  = curves.get_estimator(model_type)

        param_name ="max_depth"
        if model_type in ["Arbre de decision","Random Forest","SVM","régression logistique","xgboost classification"]:
              
                if model_type=="régression logistique":
                        param_name ="max_iter"

                result = curves.classifier_vc(X,Y,estimator,model_name,param_name)
        else:

                if model_type=="Régression Liniaire":
                        param_name ="n_jobs"
               
                result = curves.regressor_vc(X,Y,estimator,model_name,param_name)

        return jsonify(result)

@advance_alogs.route('/v1/roc_curve', methods=['POST'])
def roc_curve():
    if request.method == 'POST':

        result ={}    
        json_data = request.get_json()
        
        dataFrame,train_labels = get_dataFram(json_data)

        X = dataFrame.iloc[:,:-1] 
        Y = train_labels

        model_name  = json_data["model_name"]
        model_type  = json_data["model_type"]

        estimator  = curves.get_estimator(model_type)

        if model_type  in ["Arbre de decision","Random Forest","SVM","régression logistique","xgboost classification"]:
                
                result = curves.classifier_roc(X,Y,estimator,model_name)
    
        return jsonify(result)

@advance_alogs.route('/v1/cm_curve', methods=['POST'])
def cm_curve():
    if request.method == 'POST':

        result ={}    
        json_data = request.get_json()
        
        dataFrame,train_labels = get_dataFram(json_data)

        X = dataFrame.iloc[:,:-1] 
        Y = train_labels

        model_name  = json_data["model_name"]
        model_type  = json_data["model_type"]

        estimator  = curves.get_estimator(model_type)

        if model_type  in ["Arbre de decision","Random Forest","SVM","régression logistique","xgboost classification"]:
                
                result['link'] = curves.classifier_cm(X,Y,estimator,model_name)
    
        return jsonify(result)

@advance_alogs.route('/v1/serie', methods=['POST'])
def serie_model():
    if request.method == 'POST':

        from Models import Arima as ar

        score ={}

        json_data = request.get_json()
        
        df ,l=  ar.get_dataFram(json_data)
        test_size = json_data['trainTestValidation']["test_size"]

        df,train_set, test_set = ar.splitData(df,test_size)
        
        # if (json_data['Operation'] == "Default_Parameters"):
        clf= ar.autotuning(df, train_set,json_data['seasonal'])
        predict_set = ar.testSetPrediction(test_set,clf)

        score = ar.scoring(test_set,predict_set,clf)
        return jsonify(score)
        