from flask import Blueprint, request, jsonify
import pathlib
import os
from Models import LinearRegression as LinReg
from Models import DecisionTree as DT
import pickle
from random import randint
import pandas as  pd
import petl as etl 


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


def get_dataFram(link):
        
        tab = etl.fromcsv(link) 
        df = etl.todataframe(tab)
        train_labels=df.iloc[:,-1]

        return df,train_labels


@advance_alogs.route('/v1/tree', methods=['POST'])
def DecisionTree():
    if request.method == 'POST':
        
        json_data = request.get_json()
        link  = json_data['link']

                 # "trainTestValidation" :{
                # "test_size" :0.2,
                # "validation_size" : 0.2
                # },
                # "Operation":"Default_Parameters",
                # "link":"",
                # "parameters":{
                #         "min_samples_split":5,
                #         "max_depth ":2 
                # },

        dataFrame,train_labels = get_dataFram(link)
        model = DT
        trainTestValidation=json_data['trainTestValidation']
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
        filename = "DecisionTree_"+str(randint(0, 3000))+".pkl"

        score["filname"] = filename

        pickle.dump(clf, open(MODELS_FOLDER / filename , 'wb'))

        # Step 5: Return the response as JSON
        return jsonify(score) 



@advance_alogs.route('/v1/regression', methods=['POST'])
def regression():
    if request.method == 'POST':
        
        json_data = request.get_json()
        link  = json_data['link']

                # "trainTestValidation" :{
                # "test_size" :0.2,
                # "validation_size" : 0.2
                # },
                # "Operation":"Default_Parameters",
                # "link":"",
                # "parameters":{
                #         "fit_intercept":true,
                #         "normalize ":true ,
                #         "copy_x":true,
                #         "n_jobs":null
                # },
                #         
        dataFrame,train_labels = get_dataFram(link)
        model = LinReg
        trainTestValidation=json_data['trainTestValidation']
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
        filename = "LinearRegression_"+str(randint(0, 3000))+".pkl"

        score["filname"] = filename

        pickle.dump(clf, open(MODELS_FOLDER / filename , 'wb'))

        # Step 5: Return the response as JSON
        return jsonify(score) 


@advance_alogs.route('/v1/static', methods=['POST'])
def get_static():
    if request.method == 'POST':
        
        result =[]
        json_data = request.get_json()

        link  = json_data['link']
        fields = json_data['fields']
        tab = etl.fromcsv(link) 
        for field in fields:
                
                stats = etl.stats(tab, field)
                doc = dict(stats._asdict())
                doc["field"] = field

                result.append(doc)

        # Step 5: Return the response as JSON
        return jsonify(result) 

