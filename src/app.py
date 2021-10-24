from flask import Flask

from faiss_index.blueprint import blueprint as FaissIndexBlueprint

app = Flask(__name__)
app.config['RESOURCES_PATH'] = '/opt/faiss-instant/resources'
# app.config['RESOURCES_PATH'] = 'resources'

app.register_blueprint(FaissIndexBlueprint)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
