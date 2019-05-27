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

advance_static = Blueprint('advance_static', __name__)
  

def get_from_excel(link,sheet):

        file = requests.get(link).content
        
        df =pd.read_excel(BytesIO(file),sheet_name =sheet)
        df = df.replace(np.nan, '', regex=True)
        tab = etl.fromdataframe(df)

        return tab

def get_from_csv(link):

        file = requests.get(link).content
        
        df =pd.read_csv(BytesIO(file))

        tab = etl.fromdataframe(df) 

        return tab

@advance_static.route('/v1/static', methods=['POST'])
def get_vstatic():
    if request.method == 'POST':
        json_data = request.get_json()

        tab = {}
        if  json_data['type'] =="excel":
                tab = get_from_excel(json_data['link'],json_data['sheet'])
        elif  json_data['type'] =="csv":
                tab = get_from_csv(json_data['link'])
       
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

