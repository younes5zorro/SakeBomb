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

advance_alogs = Blueprint('advance_alogs', __name__)


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = PACKAGE_ROOT / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

MODELS_FOLDER = PACKAGE_ROOT / 'saved'
MODELS_FOLDER.mkdir(exist_ok=True)


@advance_alogs.route('/algos', methods=['GET'])
def algos():
    if request.method == 'GET':
        return jsonify({'algos_version': '0.0.0',
                        'algos_api_version': '1.1.1'})

@advance_alogs.route('/v1/clear', methods=['GET'])
def clear():
        shutil.rmtree(MODELS_FOLDER)
        MODELS_FOLDER.mkdir(exist_ok=True)
        return jsonify({"message":" removed"})

def get_dataFram(link):

        file = requests.get(link).content
        
        df =pd.read_csv(BytesIO(file))

        # tab = etl.fromcsv(link) 
        # df = etl.todataframe(tab)
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

    return model

def get_score(json_data):

        dataFrame,train_labels = get_dataFram(json_data["link"])
        trainTestValidation=json_data['trainTestValidation']
        model_name  = json_data["model_name"]
        model_type  = json_data["model_type"]
        model = get_model(model_type)

        # if(json_data['Feautue_Selection'] == True ):
        #          dataFrame=model.feature_selector(dataFrame, train_labels)

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

        if model_type =="Régression Liniaire" :
                score["x"]=X_test.reshape(1,X_test.shape[0])[0].tolist()
                score["y"]=y_test.tolist()

        joblib.dump(clf, MODELS_FOLDER / filename)
        # pickle.dump(clf, open(MODELS_FOLDER / filename , 'wb'))

        # Step 5: Return the response as JSON
        return score

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

        files = [x.name for x in MODELS_FOLDER.glob('*') if x.is_file() and  len(x.name.split('~~'))> 1 and x.name.split('~~')[2]==model_name+'.pkl']

        loaded_model = joblib.load(MODELS_FOLDER / files[0])

        df,train_labels = get_dataFram(json_data["link"])
        pred_cols = list(df.columns.values)[:-1]
        # pred_cols = list(pr.columns.values)

        # apply the whole pipeline to data
        pred = list(pd.Series(loaded_model.predict(df[pred_cols])))

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

@advance_alogs.route('/v1/static', methods=['POST'])
def get_static():
    if request.method == 'POST':
        
        result =[]
        json_data = request.get_json()

        link  = json_data['link']
        tab = etl.fromcsv(link) 

        if 'target' in json_data:

                target_key = json_data['target']
                cats = json_data['cats']
                nums = json_data['nums']

                target = etl.facet(tab, target_key)
                for key in target.keys():
                        dd = {}
                        dd["key"] = key
                        dd["data"] = []
                        for field in nums:
                                doc = {} 
                                stats = etl.stats(target[key], field)
                                doc = dict(stats._asdict())
                                doc["field"] = field
                                doc["type"] = "num"
                                dd["data"].append(doc)

                        for field in cats:
                                tt = etl.facet(target[key], field)
                                doc = {} 
                                doc["data"] = []
                                for k in tt.keys():
                                        dt ={}
                                        dt["cat"]=k
                                        dt["count"],dt["freq"] = etl.valuecount(target[key], field, k)
                                        doc["data"].append(dt)

                                doc["field"] = field
                                doc["type"] = "cat"

                                dd["data"].append(doc)
                        result.append(dd)
                
        
        else:
                fields = json_data['fields']
                for field in fields:
                        
                        stats = etl.stats(tab, field)
                        doc = dict(stats._asdict())
                        doc["field"] = field

                        result.append(doc)

        # Step 5: Return the response as JSON
        return jsonify(result) 



# @advance_alogs.route('/v1/static', methods=['POST'])
# def get_static():
#     if request.method == 'POST':
        
#         result =[]
#         json_data = request.get_json()

#         link  = json_data['link']
#         fields = json_data['fields']
#         tab = etl.fromcsv(link) 
#         for field in fields:
                
#                 stats = etl.stats(tab, field)
#                 doc = dict(stats._asdict())
#                 doc["field"] = field

#                 result.append(doc)

#         # Step 5: Return the response as JSON
#         return jsonify(result) 

