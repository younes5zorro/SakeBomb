from flask import Blueprint, request, jsonify

advance_join = Blueprint('advance_join', __name__)


@advance_join.route('/join', methods=['GET'])
def join_test():
    if request.method == 'GET':
        return jsonify({'join_test_version': '0.0.0',
                        'join_test_api_version': '1.1.1'})
