"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
from comet_ml import API
from dotenv import load_dotenv
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort
import sklearn
import pandas as pd
import joblib
import ift6758
import pickle
load_dotenv()


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
DEFAULT_MODEL_NAME = '6-LGBM.pkl'
LOADED_MODELS_DIR = os.path.join('loaded_models')

app = Flask(__name__)


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """

    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    request = {
        'workspace': "axelbogos",
        'registry_name': '6-lgbm',
        'version': '1.0.0',
    }
    if not os.path.isfile(os.path.join(LOADED_MODELS_DIR,DEFAULT_MODEL_NAME)):
        API(api_key=os.getenv('COMET_API_KEY')).download_registry_model(**request, output_path=LOADED_MODELS_DIR)
    clf = pickle.load(open(os.path.join(LOADED_MODELS_DIR,DEFAULT_MODEL_NAME), 'rb'))
    app.logger.info('Default Model Loaded!')
    print(clf)

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data
    raise NotImplementedError("TODO: implement this endpoint")

    response = None
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model
    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.
    Recommend (but not required) json with the schema:
        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
        Example of a request:
            request = {'workspace': "axelbogos",'registry_name': '6-lgbm','model_name': '6-LGBM.pkl','version': '1.0.0'}
        r = requests.post("http://0.0.0.0:8080/predict",json=request)
    """

    # Get POST json data
    json = request.get_json()
    app.logger.info(json)
    model_name = json['model_name']

    # Check to see if the model you are querying for is already downloaded
    if os.path.isfile(os.path.join(LOADED_MODELS_DIR, model_name)):
        app.logger.info(f'{model_name} is already downloaded. Loading local instance.')
        clf = pickle.load(open(os.path.join(LOADED_MODELS_DIR, model_name), 'rb'))
    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model
    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    req = {
        'workspace': json['workspace'],
        'registry_name': json['registry_name'],
        'version': json['version'],
    }

    raise NotImplementedError("TODO: implement this endpoint")

    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict
    Examples of a request:
    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    print(json)
    app.logger.info(json)

    # TODO:
    raise NotImplementedError("TODO: implement this enpdoint")
    
    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!

if __name__ == '__main__':
    app.run(port=8080)
