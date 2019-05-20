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

def get_dataFram(link,input_field,output_field):

        file = requests.get(link).content
        
        df =pd.read_csv(BytesIO(file))

        input_field.append(output_field)

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
        dataFrame,train_labels = get_dataFram(json_data["link"],json_data["input"],json_data["output"])

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

@advance_alogs.route('/v1/static', methods=['POST'])
def get_static():
    if request.method == 'POST':
        json_data = request.get_json()

        link  = json_data['link']

        file = requests.get(link).content
        
        df =pd.read_csv(BytesIO(file))

        tab = etl.fromdataframe(df) 

        cats = json_data['cats']
        stat_cat = json_data['stat_cat']
        stat_num = json_data['stat_num']
        nums = json_data['nums']

        target_key = json_data['target']

        result = {}
        result_nums  = []
        result_cats  = []
        if target_key != "" :

                for field in nums:
                        obj = {}   
                        obj["data"] =  []
                        obj["field"]=field

                        target = etl.facet(tab, target_key)
                        for key in target.keys():

                                stats_obj={}

                                stats_obj["field"]=field

                                stats_obj[target_key]=key

                                stats = etl.stats(target[key], field)

                                stats_res = dict(stats._asdict())

                                stats_res["moyenne"] = stats_res.pop("mean")
                        
                                stats_res["ecart-type"] = stats_res.pop("pstdev")
                                
                                stats_res["variance"] = stats_res.pop("pvariance")

                                for s in stat_num:
                                        stats_obj[s]=round(stats_res[s],2)

                                obj["data"].append(stats_obj)
                        
                        obj["header"] = list(obj["data"][0].keys())

                        result_nums.append(obj)

                for field in cats:

                              
                        obj = {}
                        obj["data"] =  []
                        obj["field"]=field
                        
                        target = etl.facet(tab, target_key)
                        for key in target.keys():

                                field_inst = etl.facet(target[key], field)
                                for k in field_inst.keys():

                                        stats_obj = {}
                                        stats_res ={}

                                        stats_obj[target_key]=key
                                        stats_obj[field]=k
                                        
                                        stats_res["count"],stats_res["frequence"] = etl.valuecount(target[key], field, k)
                                        for s in stat_cat:
                                                stats_obj[s]=stats_res[s]
                                        
                                        obj["data"].append(stats_obj)

                        obj["header"] = list(obj["data"][0].keys())

                        result_cats.append(obj)

        else:
                
                for field in nums:
                        
                        obj = {}   
                        obj["data"] =  []  

                        stats_obj={}

                        obj["field"]=field
                        stats_obj["field"]=field

                        stats = etl.stats(tab, field)

                        stats_res = dict(stats._asdict())

                        stats_res["moyenne"] = stats_res.pop("mean")
                
                        stats_res["ecart-type"] = stats_res.pop("pstdev")
                        
                        stats_res["variance"] = stats_res.pop("pvariance")

                        for s in stat_num:
                                stats_obj[s]=round(stats_res[s],2)

                        obj["data"].append(stats_obj)
                
                        obj["header"] = list(obj["data"][0].keys())

                        result_nums.append(obj)

                for field in cats:
                               
                        obj = {}
                        obj["data"] =  []

                        field_inst = etl.facet(tab, field)
                        for k in field_inst.keys():

                                stats_obj = {}
                                stats_res ={}

                                stats_obj[field]=k
                                obj["field"]=field
                                
                                stats_res["count"],stats_res["frequence"] = etl.valuecount(tab, field, k)
                                for s in stat_cat:
                                        stats_obj[s]=stats_res[s]
                                
                                obj["data"].append(stats_obj)

                        obj["header"] = list(obj["data"][0].keys())

                        result_cats.append(obj)

        result["cat"]=result_cats
        result["num"]=result_nums

        return jsonify(result)

@advance_alogs.route('/v1/tt', methods=['POST'])
def ttttt():
        
        
 
        if target_key != "" :

                

                        if len(nums) > 0:

                                catss = {}
                                catss["data"] =  []

                                for field in nums:

                                        gg={}

                                        gg[target_key]=key
                                        catss["field"]=field
                                        gg["field"]=field

                                        stats = etl.stats(target[key], field)

                                        dd = dict(stats._asdict())

                                        dd["moyenne"] = dd.pop("mean")
                                
                                        dd["ecart-type"] = dd.pop("pstdev")
                                        
                                        dd["variance"] = dd.pop("pvariance")
                                        
                                        for s in stat_num:
                                               gg[s]=round(dd[s],2)

                                        catss["data"].append(gg)

                                catss["header"] = list(catss["data"][0].keys())
                                added = True
                                # for item in result_nums :
                                #         if item["header"] == catss["header"]:
                                #                 item["data"].extend(catss["data"])
                                #                 added = False
                                                
                                if added : result_nums.append(catss)

                        if len(cats) > 0:

                           for field in cats:
                                
                                catss = {}
                                catss["data"] =  [] 
                                tt = etl.facet(target[key], field)

                                for k in tt.keys():

                                        gg = {}
                                        gg[target_key]=key

                                        gg[field]=k
                                        # gg[field]=k
                                        catss["field"]=field

                                        dd ={}
                                        dd["count"],dd["frequence"] = etl.valuecount(target[key], field, k)


                                        for s in stat_cat:
                                                gg[s] = round(dd[s],2)

                                       
                                        catss["data"].append(gg)

                                
                                catss["header"] = list(catss["data"][0].keys())
                                
                                added = True
                                for item in result_cats :
                                        if item["header"] == catss["header"]:
                                                item["data"].extend(catss["data"])
                                                added = False
                                                
                                if added : result_cats.append(catss)

                        result["cat"]=result_cats
                        result["num"]=result_nums        
        
        else:
                
 
                if len(nums) > 0:

                        catss = {}
                        catss["data"] =  []

                        for field in nums:

                                gg={}

                                catss["field"]=field
                                gg["field"]=field
                                stats = etl.stats(tab, field)

                                dd = dict(stats._asdict())

                                dd["moyenne"] = dd.pop("mean")
                        
                                dd["ecart-type"] = dd.pop("pstdev")
                                
                                dd["variance"] = dd.pop("pvariance")

                                for s in stat_num:
                                        gg[s]=round(dd[s],2)

                                
                                        
                                # if "sum" in gg:
                                #         gg["somme"] = gg.pop("sum")
                                        
                                # if "freq" in gg:
                                #         gg["frequence"] = gg.pop("freq")

                                        
                                catss["data"].append(gg)
                        catss["header"] = list(catss["data"][0].keys())

                        result_nums.append(catss)
                        
                if len(cats) > 0:

                        for field in cats:
                               
                                catss = {}
                                catss["data"] =  []

                                tt = etl.facet(tab, field)
                                for k in tt.keys():

                                        gg = {}
                                        gg[field]=k
                                        catss["field"]=field
                                        dd ={}

                                        dd["count"],dd["frequence"] = etl.valuecount(tab, field, k)
                                        for s in stat_cat:
                                                gg[s]=dd[s]
                                        
                                        catss["data"].append(gg)

                                catss["header"] = list(catss["data"][0].keys())

                                result_cats.append(catss)

                result["cat"]=result_cats
                result["num"]=result_nums


        # Step 5: Return the response as JSON
        return jsonify(result) 

# @advance_alogs.route('/v1/static', methods=['POST'])
# def get_static():
#     if request.method == 'POST':
        
#         json_data = request.get_json()

#         link  = json_data['link']

#         file = requests.get(link).content
        
#         df =pd.read_csv(BytesIO(file))

#         tab = etl.fromdataframe(df) 

#         cats = json_data['cats']
#         stat_cat = json_data['stat_cat']
#         stat_num = json_data['stat_num']
#         nums = json_data['nums']

#         target_key = json_data['target']

#         result = {}
#         result_nums  = []
#         result_cats  = []

#         if target_key != "" :

#                 target = etl.facet(tab, target_key)
#                 for key in target.keys():

#                         if len(nums) > 0:

#                                 catss = {}
#                                 catss["data"] =  []
#                                 for field in nums:

#                                         gg={}

#                                         gg[target_key]=key
#                                         catss["field"]=field
#                                         gg["field"]=field

#                                         stats = etl.stats(target[key], field)

#                                         dd = dict(stats._asdict())

#                                         dd["moyenne"] = dd.pop("mean")
                                
#                                         dd["ecart-type"] = dd.pop("pstdev")
                                        
#                                         dd["variance"] = dd.pop("pvariance")
                                        
#                                         for s in stat_num:
#                                                gg[s]=round(dd[s],2)

#                                         catss["data"].append(gg)

#                                 catss["header"] = list(catss["data"][0].keys())
#                                 added = True
#                                 # for item in result_nums :
#                                 #         if item["header"] == catss["header"]:
#                                 #                 item["data"].extend(catss["data"])
#                                 #                 added = False
                                                
#                                 if added : result_nums.append(catss)

#                         if len(cats) > 0:

#                            for field in cats:
                                
#                                 catss = {}
#                                 catss["data"] =  [] 
#                                 tt = etl.facet(target[key], field)

#                                 for k in tt.keys():

#                                         gg = {}
#                                         gg[target_key]=key

#                                         gg[field]=k
#                                         # gg[field]=k
#                                         catss["field"]=field

#                                         dd ={}
#                                         dd["count"],dd["frequence"] = etl.valuecount(target[key], field, k)


#                                         for s in stat_cat:
#                                                 gg[s] = round(dd[s],2)

                                       
#                                         catss["data"].append(gg)

                                
#                                 catss["header"] = list(catss["data"][0].keys())
                                
#                                 added = True
#                                 for item in result_cats :
#                                         if item["header"] == catss["header"]:
#                                                 item["data"].extend(catss["data"])
#                                                 added = False
                                                
#                                 if added : result_cats.append(catss)

#                         result["cat"]=result_cats
#                         result["num"]=result_nums        
        
#         else:
                
 
#                 if len(nums) > 0:

#                         catss = {}
#                         catss["data"] =  []

#                         for field in nums:

#                                 gg={}

#                                 catss["field"]=field
#                                 gg["field"]=field
#                                 stats = etl.stats(tab, field)

#                                 dd = dict(stats._asdict())

#                                 dd["moyenne"] = dd.pop("mean")
                        
#                                 dd["ecart-type"] = dd.pop("pstdev")
                                
#                                 dd["variance"] = dd.pop("pvariance")

#                                 for s in stat_num:
#                                         gg[s]=round(dd[s],2)

                                
                                        
#                                 # if "sum" in gg:
#                                 #         gg["somme"] = gg.pop("sum")
                                        
#                                 # if "freq" in gg:
#                                 #         gg["frequence"] = gg.pop("freq")

                                        
#                                 catss["data"].append(gg)
#                         catss["header"] = list(catss["data"][0].keys())

#                         result_nums.append(catss)
                        
#                 if len(cats) > 0:

#                         for field in cats:
                               
#                                 catss = {}
#                                 catss["data"] =  []

#                                 tt = etl.facet(tab, field)
#                                 for k in tt.keys():

#                                         gg = {}
#                                         gg[field]=k
#                                         catss["field"]=field
#                                         dd ={}

#                                         dd["count"],dd["frequence"] = etl.valuecount(tab, field, k)
#                                         for s in stat_cat:
#                                                 gg[s]=dd[s]
                                        
#                                         catss["data"].append(gg)

#                                 catss["header"] = list(catss["data"][0].keys())

#                                 result_cats.append(catss)

#                 result["cat"]=result_cats
#                 result["num"]=result_nums


#         # Step 5: Return the response as JSON
#         return jsonify(result) 

@advance_alogs.route('/v1/selection', methods=['POST'])
def features_selection():
        if request.method == 'POST':
                from sklearn.linear_model import Lasso
                from sklearn.feature_selection import SelectFromModel

                json_data = request.get_json()

                result ={}
                X_train, y_train= get_ForSelection(json_data["link"],json_data["input"],json_data["output"])

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
                link = json_data["link"]
                header = json_data["header"]

                file = requests.get(link).content
        
                df =pd.read_csv(BytesIO(file))

                df = df[header]

                result = df.values.tolist()

                return jsonify(result)

def get_ForSelection(link,input_field,output_field):

        file = requests.get(link).content
        
        df ,l =get_dataFram(link,input_field,output_field)

        X_train = df.iloc[:,:-1]  #independent columns
        y_train = df.iloc[:,-1] 

        return X_train,y_train

@advance_alogs.route('/v1/l_curve', methods=['POST'])
def l_curve():
    if request.method == 'POST':

        result ={}    
        json_data = request.get_json()
        
        dataFrame,train_labels = get_dataFram(json_data["link"],json_data["input"],json_data["output"])

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
        
        dataFrame,train_labels = get_dataFram(json_data["link"],json_data["input"],json_data["output"])

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
        
        dataFrame,train_labels = get_dataFram(json_data["link"],json_data["input"],json_data["output"])

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
        
        dataFrame,train_labels = get_dataFram(json_data["link"],json_data["input"],json_data["output"])

        X = dataFrame.iloc[:,:-1] 
        Y = train_labels

        model_name  = json_data["model_name"]
        model_type  = json_data["model_type"]

        estimator  = curves.get_estimator(model_type)

        if model_type  in ["Arbre de decision","Random Forest","SVM","régression logistique","xgboost classification"]:
                
                result['link'] = curves.classifier_cm(X,Y,estimator,model_name)
    
        return jsonify(result)


@advance_alogs.route('/v1/backstatic', methods=['POST'])
def get_backstatic():
    if request.method == 'POST':
        
        json_data = request.get_json()

        link  = json_data['link']

        file = requests.get(link).content
        
        df =pd.read_csv(BytesIO(file))

        tab = etl.fromdataframe(df) 

        cats = json_data['cats']
        stat_cat = json_data['stat_cat']
        stat_num = json_data['stat_num']
        nums = json_data['nums']
        

        ntarget_key = json_data['ntarget']
        ctarget_key = json_data['ctarget']

        result = {}
        result_nums  = []
        result_cats  = []
 
        
        if ntarget_key == "":

                if len(nums) > 0:

                        catss = {}
                        catss["data"] =  []

                        for field in nums:

                                gg={}

                                catss["field"]=field
                                gg["field"]=field
                                stats = etl.stats(tab, field)

                                dd = dict(stats._asdict())

                                dd["moyenne"] = dd.pop("mean")
                        
                                dd["ecart-type"] = dd.pop("pstdev")
                                
                                dd["variance"] = dd.pop("pvariance")

                                for s in stat_num:
                                        gg[s]=round(dd[s],2)

                                catss["data"].append(gg)
                                
                        catss["header"] = list(catss["data"][0].keys())

                        result_nums.append(catss)

        else:
                target = etl.facet(tab, ntarget_key)
                for key in target.keys():
                        if len(nums) > 0:

                                catss = {}
                                catss["data"] =  []
                                for field in nums:

                                        gg={}

                                        gg[ntarget_key]=key
                                        catss["field"]=field
                                        gg["field"]=field

                                        stats = etl.stats(target[key], field)

                                        dd = dict(stats._asdict())

                                        dd["moyenne"] = dd.pop("mean")
                                
                                        dd["ecart-type"] = dd.pop("pstdev")
                                        
                                        dd["variance"] = dd.pop("pvariance")
                                        
                                        for s in stat_num:
                                               gg[s]=round(dd[s],2)


                                        catss["data"].append(gg)

                                catss["header"] = list(catss["data"][0].keys())
                                added = True
                                for item in result_nums :
                                        if item["header"] == catss["header"]:
                                                item["data"].extend(catss["data"])
                                                added = False
                                                
                                if added : result_nums.append(catss)
        
        if ctarget_key == "":

                if len(cats) > 0:

                        for field in cats:
                               
                                catss = {}
                                catss["data"] =  []

                                tt = etl.facet(tab, field)
                                for k in tt.keys():

                                        gg = {}
                                        gg[field]=k
                                        catss["field"]=field
                                        dd ={}

                                        dd["count"],dd["frequence"] = etl.valuecount(tab, field, k)
                                        for s in stat_cat:
                                                gg[s]=dd[s]
                                        
                                        catss["data"].append(gg)

                                catss["header"] = list(catss["data"][0].keys())

                                result_cats.append(catss)

        else:

                target = etl.facet(tab, ctarget_key)
                for key in target.keys():

                        if len(cats) > 0:

                           for field in cats:
                                
                                catss = {}
                                catss["data"] =  [] 
                                tt = etl.facet(target[key], field)

                                for k in tt.keys():

                                        gg = {}
                                        gg[ctarget_key]=key
                                        gg[field]=k
                                        catss["field"]=field

                                        dd ={}
                                        dd["count"],dd["frequence"] = etl.valuecount(target[key], field, k)


                                        for s in stat_cat:
                                                gg[s] = round(dd[s],2)

                                       
                                        catss["data"].append(gg)

                                
                                catss["header"] = list(catss["data"][0].keys())
                                
                                added = True
                                for item in result_cats :
                                        if item["header"] == catss["header"]:
                                                item["data"].extend(catss["data"])
                                                added = False
                                                
                                if added : result_cats.append(catss)

        result["cat"]=result_cats
        result["num"]=result_nums     
                

        # Step 5: Return the response as JSON
        return jsonify(result) 

# @advance_alogs.route('/v1/static', methods=['POST'])
# def get_static():
#     if request.method == 'POST':
        
#         json_data = request.get_json()

#         link  = json_data['link']

#         file = requests.get(link).content
        
#         df =pd.read_csv(BytesIO(file))

#         tab = etl.fromdataframe(df) 

#         cats = json_data['cats']
#         stat_cat = json_data['stat_cat']
#         stat_num = json_data['stat_num']
#         nums = json_data['nums']

#         target_key = json_data['target']

#         if target_key != "" :
#                 result =[]

#                 target = etl.facet(tab, target_key)
#                 for key in target.keys():
#                         dd = {}
#                         dd["key"] = key
#                         dd["data"] = []

#                         if len(nums) > 0:

#                                 for field in nums:
#                                         doc = {} 
#                                         stats = etl.stats(target[key], field)

#                                         ff = dict(stats._asdict())

#                                         for s in stat_num:
#                                                 doc[s] = ff[s]
#                                         doc["field"] = field
#                                         # doc["type"] = "num"
#                                         dd["data"].append(doc)
#                                 result.append(dd)
#                         else:
#                            for field in cats:
#                                 tt = etl.facet(target[key], field)
#                                 doc = {} 
#                                 doc["data"] = []
#                                 for k in tt.keys():
#                                         dt ={}
#                                         ff ={}
#                                         dt["cat"]=k
#                                         ff["count"],ff["freq"] = etl.valuecount(target[key], field, k)
#                                         for s in stat_cat:
#                                                 dt[s] = ff[s]
#                                         doc["data"].append(dt)

#                                 doc["field"] = field
#                                 # doc["type"] = "cat"

#                                 dd["data"].append(doc)
#                            result.append(dd)
        
#         else:
#                 result = {}
#                 result_nums  = []
#                 result_cats  = []
 
#                 if len(nums) > 0:
#                         table = {}
#                         table["field"] = []

#                         for field in nums:
#                                 # doc = {} 
#                                 stats = etl.stats(tab, field)

#                                 dd = dict(stats._asdict())
#                                 for s in stat_num:
#                                         if field == nums[0]:
#                                                 table[s] = []

#                                         table[s].append(dd[s])

#                                 table["field"].append(field)
#                         result_nums.append(table)
                        
#                 if len(cats) > 0:

#                         for field in cats:
                               
#                                 catss = {}
#                                 table = {}

#                                 table[field] = []
#                                 for s in stat_cat:
#                                         table[s] = []

                                
#                                 tt = etl.facet(tab, field)
#                                 # doc = {} 
#                                 # doc["data"] = []
#                                 for k in tt.keys():
#                                         table[field].append(k)
#                                         # dt ={}
#                                         dd ={}
#                                         # dt["cat"]=k
#                                         dd["count"],dd["freq"] = etl.valuecount(tab, field, k)
#                                         for s in stat_cat:
                                                
#                                                 table[s].append(dd[s])

#                                         # doc["data"].append(dt)

#                                 # doc["field"] = field
#                                 # doc["type"] = "cat"

#                                 if "freq" in table:
#                                      table["frequence"] = table.pop("freq")

#                                 result_cats.append(table)

#                 result["cat"]=result_cats
#                 result["num"]=result_nums


#         # Step 5: Return the response as JSON
#         return jsonify(result) 

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
