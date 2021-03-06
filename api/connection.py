from flask import Blueprint, request, jsonify
from sqlalchemy import create_engine
import openpyxl

import requests
import pandas as pd
import numpy as np
from io import BytesIO
import petl as etl 
import json

advance_app = Blueprint('advance_app', __name__)

def get_from_excel(link,sheet):

        file = requests.get(link).content
        
        df =pd.read_excel(BytesIO(file),sheet_name =sheet)
        df = df.replace(np.nan, '', regex=True)

        return df

def get_from_csv(link):

        file = requests.get(link).content
        
        df =pd.read_csv(BytesIO(file))

        return df


@advance_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': '0.0.0',
                        'api_version': '1.1.1'})

@advance_app.route('/v1/connect', methods=['POST'])
def connect_db():
    if request.method == 'POST':
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()

        result = {}
        _PORT  = json_data['port']
        _HOST  = json_data['host']
        _USER  = json_data['username']
        _PASS  = json_data['password']
        _DATABASE  = json_data['db_name']

        
        engine = create_engine('mysql+pymysql://'+_USER+':'+_PASS+'@'+_HOST+':'+_PORT)

        if(engine.connect()):
                result["dbs"] = [db for (db,) in engine.execute("show databases") ]
                if(_DATABASE != ""  ):
                        if (_DATABASE in result["dbs"]):
                                result["status"] = True
                                engine.connect().execute("USE "+_DATABASE +" ;")
                                result["tabs"] = [tab for (tab,) in engine.execute("SHOW TABLES;")   ]
                        else:
                                result["status"] = False
                                result["error"] = "Database not in Server"
        else:
                result["status"] = True
                result["error"] = "Connexion failed"

        # # Step 5: Return the response as JSON
        return jsonify(result) 

@advance_app.route('/v1/file', methods=['POST'])
def connect_file():
    if request.method == 'POST':
        
        json_data = request.get_json()
        link  = json_data['link']

        file = requests.get(link).content
        
        # Step 1: check if the post request has the file part
        # if 'file' not in request.files:
        #     return jsonify('No file found'), 400

        # file = request.files['file']

        # filename = secure_filename(file.filename)

        #     # Step 3: Save the file
        #     # Note, in production, this would require careful
        #     # validation, management and clean up.
        # file.save(os.path.join(UPLOAD_FOLDER, filename))

        result = {}
        wb = openpyxl.load_workbook(filename=BytesIO(file))
        result["sheets"] = wb.get_sheet_names()

        # Step 5: Return the response as JSON
        return jsonify(result) 

@advance_app.route('/v1/data/file', methods=['POST'])
def data_excel():
    if request.method == 'POST':
        result = {}
        df ={}
        
        json_data = request.get_json()

        link  = json_data['link']
        sheet  = json_data['sheet']
        type_file =json_data['type']

        if  type_file =="excel":
                result["sheet_name"] = sheet
                df = get_from_excel(link,sheet)

        elif  type_file =="csv":
                df = get_from_csv(link)
        
        # file = requests.get(link).content
        

        # df =pd.read_excel(BytesIO(file),sheet_name =sheet)
        # df = df.replace(np.nan, '', regex=True)
        tab = etl.fromdataframe(df)
        # if tab:
        result["header"] = list(etl.header(tab))
        result["data"] =  list(etl.dicts(tab))

        result["categorical"] = []
        result["numerical"] = []
        
        for var in df.columns:

                if df[var].dtypes=='O':
                        result["categorical"].append(var)
                else:
                        # if len(df[var].unique())<20:
                        #        result["categorical"].append(var)
                        # else:
                                result["numerical"].append(var)
        # else:
        #         result["sheet_name"] = sheet
        #         result["header"] = []
        #         result["data"] =  []
        #         result["categorical"] = []
        #         result["numerical"] = []
        
        # Step 5: Return the response as JSON
        return jsonify(result) 

@advance_app.route('/v1/data/file/csv', methods=['POST'])
def data_csv():
    if request.method == 'POST':

        
        result = {}
        
        json_data = request.get_json()
        link  = json_data['link']

        file = requests.get(link).content
        
        df =pd.read_csv(BytesIO(file))

        tab = etl.fromdataframe(df) 


        result["categorical"] = []
        result["numerical"] = []
        
        for var in df.columns:

                if df[var].dtypes=='O':
                        result["categorical"].append(var)
                else:
                        # if len(df[var].unique())<20:
                        #        result["categorical"].append(var)
                        # else:
                                result["numerical"].append(var)
                                
        result["header"]  = list(etl.header(tab))

        result["data"] = list(etl.dicts(tab))

        # Step 5: Return the response as JSON
        return jsonify(result) 

@advance_app.route('/v1/data/db', methods=['POST'])
def data_db():
    if request.method == 'POST':
        
        json_data = request.get_json()

        result = {}
        _PORT  = json_data['port']
        _HOST  = json_data['host']
        _USER  = json_data['username']
        _PASS  = json_data['password']
        _DATABASE  = json_data['db_name']
        _TABLE  = json_data['table']

        engine = create_engine('mysql+pymysql://'+_USER+':'+_PASS+'@'+_HOST+':'+_PORT)

        # rows = engine.execute('SELECT * FROM '+_DATABASE+'.'+_TABLE)

        df = pd.read_sql('SELECT * FROM '+_DATABASE+'.'+_TABLE, con=engine)
        tab = etl.fromdataframe(df) 

        # data =[dict(row) for row in rows]

        # tab  = etl.fromdicts(data)
        result["categorical"] = []
        result["numerical"] = []

        result["header"] = list(etl.header(tab))
        result["data"] = list(etl.dicts(tab))
        
        for var in df.columns:

                if df[var].dtypes=='O':
                        result["categorical"].append(var)
                else:
                        # if len(df[var].unique())<20:
                        #        result["categorical"].append(var)
                        # else:
                                result["numerical"].append(var)

        # # Step 5: Return the response as JSON
        return jsonify(result) 

