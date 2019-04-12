from flask import Blueprint, request, jsonify
from sqlalchemy import create_engine
import openpyxl

import requests

from io import BytesIO
import petl as etl 


advance_app = Blueprint('advance_app', __name__)


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
        _TYPE  = json_data['type']
        _PORT  = json_data['port']
        _HOST  = json_data['host']
        _USER  = json_data['user']
        _PASS  = json_data['pass']
        _DATABASE  = json_data['database']

        
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
def data_file():
    if request.method == 'POST':
        
        json_data = request.get_json()
        link  = json_data['link']
        sheet  = json_data['sheet']

        file = requests.get(link).content
        
      
        result = {}
        excel_file = openpyxl.load_workbook(filename=BytesIO(file))

        if sheet not in excel_file.get_sheet_names():
                result["status"] =  False
                result["Error"] =  "Sheet not in the file "
                return jsonify(result) 

        tab = etl.fromxlsx(filename=BytesIO(file),sheet =sheet)
        if tab:
                result["sheet_name"] = sheet
                result["header"] = list(etl.header(tab))
                result["data"] =  list(etl.dicts(tab))
                result["status"] =  True

        else:
                result["Error"] =  "Sheet Empty"
                result["status"] =  False

        # Step 5: Return the response as JSON
        return jsonify(result) 


@advance_app.route('/v1/data/db', methods=['POST'])
def data_db():
    if request.method == 'POST':
        
        json_data = request.get_json()

        result = {}
        _TYPE  = json_data['type']
        _PORT  = json_data['port']
        _HOST  = json_data['host']
        _USER  = json_data['user']
        _PASS  = json_data['pass']
        _DATABASE  = json_data['database']
        _TABLE  = json_data['table']

        engine = create_engine('mysql+pymysql://'+_USER+':'+_PASS+'@'+_HOST+':'+_PORT)

        if(engine.connect()):
                result["dbs"] = [db for (db,) in engine.execute("show databases") ]
                if(_DATABASE != ""  ):
                        if (_DATABASE in result["dbs"]):
                                result["status"] = True
                                engine.connect().execute("USE "+_DATABASE +" ;")
                                result["tabs"] = [tab for (tab,) in engine.execute("SHOW TABLES;")   ]

                                if(_TABLE in result["tabs"]):   
                                        rows = engine.execute('SELECT * FROM '+_DATABASE+'.'+_TABLE +' LIMIT 10')

                                        data =[dict(row) for row in rows]

                                        tab  = etl.fromdicts(data)
                                        
                                        result["header"] = list(etl.header(tab))
                                        result["data"] = list(etl.dicts(tab))
                                else:
                                        result["status"] = False
                                        result["error"] = "Table not in database"
                        else:
                                result["status"] = False
                                result["error"] = "Database not in Server"
                else:
                        result["status"] = False
                        result["error"] = "Not database specified"
        else:
                result["status"] = False
                result["error"] = "Connexion failed"

        # # Step 5: Return the response as JSON
        return jsonify(result) 


# @advance_app.route('/v1/predict/regression', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Step 1: Extract POST data from request body as JSON
#         json_data = request.get_json()
#         _logger.debug(f'Inputs: {json_data}')

#         # Step 2: Validate the input using marshmallow schema
#         input_data, errors = validate_inputs(input_data=json_data)

#         # Step 3: Model prediction
#         result = make_prediction(input_data=input_data)
#         _logger.debug(f'Outputs: {result}')

#         # Step 4: Convert numpy ndarray to list
#         predictions = result.get('predictions').tolist()
#         version = result.get('version')

#         # Step 5: Return the response as JSON
#         return jsonify({'predictions': predictions,
#                         'version': version,
#                         'errors': errors})
