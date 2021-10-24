from jsonschema import validate, ValidationError
from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest
from faiss_index import FaissIndex

blueprint = Blueprint('faiss_index', __name__)

@blueprint.record_once
def record(setup_state):
    blueprint.faiss_index = FaissIndex(setup_state.app.config.get('RESOURCES_PATH'))

@blueprint.route('/search', methods=['POST'])
def search():
    try:
        json = request.get_json(force=True)
        validate(json, {
            'type': 'object',
            'required': ['k', 'vectors'],
            'properties': {
                'k': { 'type': 'integer', 'minimum': 1 },
                'vectors': {
                    'type': 'array',
                    'items': {
                        'type': 'array',
                        'items': { 'type': 'number' }
                    }
                }
            }
        })
        if 'vectors' not in json:
            return jsonify([])
        
        results = blueprint.faiss_index.search(json['vectors'], json['k'])
        return jsonify(results)

    except (BadRequest, ValidationError) as e:
        print('Bad request', e)
        return 'Bad request', 400

    except Exception as e:
        print('Server error', e)
        return 'Server error', 500

@blueprint.route('/reload', methods=['GET'])
def reload():
    blueprint.faiss_index.load()
    return 'Faiss index reloaded\n', 200