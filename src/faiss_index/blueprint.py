from jsonschema import validate, ValidationError
from flask import Blueprint, jsonify, request, current_app
from werkzeug.exceptions import BadRequest
from faiss_index import FaissIndex
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

blueprint = Blueprint('faiss_index', __name__)

@blueprint.record_once
def record(setup_state):
    resources_path = setup_state.app.config.get('RESOURCES_PATH')
    use_gpu = setup_state.app.config.get('USE_GPU')
    blueprint.faiss_index = FaissIndex(resources_path, logger, use_gpu)

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
        logger.info('Bad request', e)
        return 'Bad request', 400

    except Exception as e:
        logger.info('Server error', e)
        return 'Server error', 500

@blueprint.route('/explain', methods=['POST'])
def explain():
    try:
        json = request.get_json(force=True)
        validate(json, {
            'type': 'object',
            'required': ['vector', 'id'],
            'properties': {
                'id': { 'type': 'string', 'minimum': 1 },
                'vector': {
                    'type': 'array',
                    'items': {
                        'type': 'number',
                    }
                }
            }
        })
        result = blueprint.faiss_index.explain(json['vector'], json['id'])
        return jsonify(result)

    except (BadRequest, ValidationError) as e:
        logger.info('Bad request', e)
        return 'Bad request', 400

    except Exception as e:
        logger.info('Server error', e)
        return 'Server error', 500

@blueprint.route('/reload', methods=['POST'])
def reload():
    data = request.get_json()
    index_name = None
    use_gpu = False
    if data is not None:
        assert 'index_name' in data, "Please specify 'index_name' in the URL arguments."
        index_name = data['index_name']
        use_gpu = data.setdefault('use_gpu', False)
    blueprint.faiss_index.load(index_name, use_gpu)
    return f'Faiss index ({index_name}) reloaded\n', 200

@blueprint.route('/index_list', methods=['GET'])
def index_list():
    index_list = blueprint.faiss_index.parse_index_list()
    index_loaded = blueprint.faiss_index.index_loaded
    device = blueprint.faiss_index.device
    results = {
        'index loaded': index_loaded,
        'device': device,
        'index list': index_list
    }
    return jsonify(results)

@blueprint.route('/reconstruct', methods=['GET'])
def reconstruct():
    _id = request.args.get('id')
    result = {'vector': blueprint.faiss_index.reconstruct(_id).tolist()}
    return jsonify(result)