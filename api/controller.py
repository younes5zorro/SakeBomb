from flask import Blueprint, request, jsonify
from sqlalchemy import create_engine


advance_app = Blueprint('advance_app', __name__)


@advance_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': '0.0.0',
                        'api_version': '1.1.1'})


@advance_app.route('/v1/connect', methods=['POST'])
def connect():
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


# @advance_app.route('/predict/classifier', methods=['POST'])
# def predict_image():
#     if request.method == 'POST':
#         # Step 1: check if the post request has the file part
#         if 'file' not in request.files:
#             return jsonify('No file found'), 400

#         file = request.files['file']

#         # Step 2: Basic file extension validation
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)

#             # Step 3: Save the file
#             # Note, in production, this would require careful
#             # validation, management and clean up.
#             file.save(os.path.join(UPLOAD_FOLDER, filename))

#             _logger.debug(f'Inputs: {filename}')

#             # Step 4: perform prediction
#             result = make_single_prediction(
#                 image_name=filename,
#                 image_directory=UPLOAD_FOLDER)

#             _logger.debug(f'Outputs: {result}')

#         readable_predictions = result.get('readable_predictions')
#         version = result.get('version')

#         # Step 5: Return the response as JSON
#         return jsonify(
#             {'readable_predictions': readable_predictions[0],
#              'version': version})
