from flask import Blueprint, request, jsonify
from sqlalchemy import create_engine
import openpyxl

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
                result["status"] = True
                result["dbs"] = [db for (db,) in engine.execute("show databases") ]
                if(_DATABASE != "" and _DATABASE in result["dbs"]):
                        engine.connect().execute("USE "+_DATABASE +" ;")
                        result["tabs"] = [tab for (tab,) in engine.execute("SHOW TABLES;")   ]


        # # Step 5: Return the response as JSON
        return jsonify(result) 


@advance_app.route('/v1/file', methods=['POST'])
def connect_file():
    if request.method == 'POST':

        print(request.files)
        # Step 1: check if the post request has the file part
        if 'file' not in request.files:
            return jsonify('No file found'), 400

        file = request.files['file']

        result = {}
        wb = openpyxl.load_workbook(file)
        result["sheets"] = wb.get_sheet_names()

        # Step 5: Return the response as JSON
        return jsonify(result) 

