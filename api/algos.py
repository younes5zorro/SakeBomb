from flask import Blueprint, request, jsonify
import pathlib
import os
from Models import LinearRegression as LinReg
import pickle
from random import randint
import pandas as  pd


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


def get_dataFram():
        
        dataFrame = pd.read_csv(UPLOAD_FOLDER / "flights.csv")
        train_labels=dataFrame.iloc[:,-1]

        return dataFrame,train_labels


@advance_alogs.route('/v1/regression', methods=['POST'])
def join():
    if request.method == 'POST':
        
        json_data = request.get_json()

                # "trainTestValidation" :{
                # "test_size" :0.2,
                # "validation_size" : 0.2
                # },
                # "Operation":"Default_Parameters",
                # "parameters":{
                #         "fit_intercept":true,
                #         "normalize ":true ,
                #         "copy_x":true,
                #         "n_jobs":null
                # },

        dataFrame,train_labels = get_dataFram()
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

        filename = "LinearRegression_"+str(randint(0, 9))+".pkl"
        pickle.dump(clf, open(MODELS_FOLDER / filename , 'wb'))

        # Step 5: Return the response as JSON
        return jsonify(score) 

