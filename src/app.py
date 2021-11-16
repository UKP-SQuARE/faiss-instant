from flask import Flask
from faiss_index.blueprint import blueprint as FaissIndexBlueprint
import argparse

def cli_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help="Only for source code debugging. Resource folder './resource' will be used.")
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU or not.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = cli_arguments()

    app = Flask(__name__)
    if args.debug:
        app.config['RESOURCES_PATH'] = 'resources'        
        port = 5001
    else:
        app.config['RESOURCES_PATH'] = '/opt/faiss-instant/resources'
        port = 5000
    
    if args.use_gpu:
        app.config['USE_GPU'] = True
    else:
        app.config['USE_GPU'] = False

    app.register_blueprint(FaissIndexBlueprint)
    app.run(host='0.0.0.0', port=port, debug=args.debug)
