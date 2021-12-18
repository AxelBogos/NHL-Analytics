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
CLASSIFIER = None
app = Flask(__name__)


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    global CLASSIFIER  # make this variable global to the scope of the app

    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    request = {
        'workspace': "axelbogos",
        'registry_name': '6-lgbm',
        'version': '1.0.0',
    }

    if not os.path.isfile(os.path.join(LOADED_MODELS_DIR, DEFAULT_MODEL_NAME)):
        API(api_key=os.getenv('COMET_API_KEY')).download_registry_model(**request, output_path=LOADED_MODELS_DIR)
    CLASSIFIER = pickle.load(open(os.path.join(LOADED_MODELS_DIR, DEFAULT_MODEL_NAME), 'rb'))
    app.logger.info('Default Model Loaded!')


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response

    Example:
        r = requests.get("http://0.0.0.0:8080/logs")

    """
    with open('flask.log') as f:
        response = f.read()
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model
    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.
    Examples of requests:
        LGBM:
            import requests
            request = {'workspace': "axelbogos",'registry_name': '6-lgbm','model_name': '6-LGBM.pkl','version': '1.0.0'}
            r = requests.post("http://0.0.0.0:5000/download_registry_model",json=request)
        5-2 XGB:
            request = {'workspace': "axelbogos",'registry_name': '5-2-grid-search-model','model_name': 'tuned_xgb_model.pkl','version': '2.0.0'}
            r = requests.post("http://0.0.0.0:5000/download_registry_model",json=request)
        5-3 XGB:
            request = {'workspace': "axelbogos",'registry_name': '5-3-best-feature','model_name': 'xgb_feature.pkl','version': '1.0.0'}
            r = requests.post("http://0.0.0.0:5000/download_registry_model",json=request)
        6-2 NN:
            request = {'workspace': "axelbogos",'registry_name': '6-2-nn-tuned-model','model_name': 'tuned_nn_model.pkl','version': '1.0.0'}
            r = requests.post("http://0.0.0.0:5000/download_registry_model",json=request)
        6-3 Adaboost:
            request = {'workspace': "axelbogos",'registry_name': '6-3-adaboost-tuned-model','model_name': 'tuned_adaboost_model.pkl','version': '1.0.0'}
            r = requests.post("http://0.0.0.0:5000/download_registry_model",json=request)
        6-4 Stacked Classifier:
            request = {'workspace': "axelbogos",'registry_name': '6-4-stacked-trained-tuned-model','model_name': 'tuned_stacked_trained_model.pkl','version': '1.0.0'}
            r = requests.post("http://0.0.0.0:5000/download_registry_model",json=request)
    """

    global CLASSIFIER  # make this variable global to the scope of the app

    # Get POST json data
    json = request.get_json()
    app.logger.info(json)
    model_name = json['model_name']

    # Check to see if the model you are querying for is already downloaded
    if os.path.isfile(os.path.join(LOADED_MODELS_DIR, model_name)):
        app.logger.info(f'{model_name} is already downloaded. Loading local instance.')
        CLASSIFIER = pickle.load(open(os.path.join(LOADED_MODELS_DIR, model_name), 'rb'))
        response = 'Success'
    else:
        # Make API request
        req = {
            'workspace': json['workspace'],
            'registry_name': json['registry_name'],
            'version': json['version'],
        }
        # Request the API
        API(api_key=os.getenv('COMET_API_KEY')).download_registry_model(**req, output_path=LOADED_MODELS_DIR)

        # check if success
        if os.path.isfile(os.path.join(LOADED_MODELS_DIR, DEFAULT_MODEL_NAME)):
            CLASSIFIER = pickle.load(open(os.path.join(LOADED_MODELS_DIR, model_name), 'rb'))
            app.logger.info(f'Download successful. Loaded {model_name}.')
            response = 'Success'
        else:
            app.logger.info(f'Download failed. Current model loaded is {str(CLASSIFIER)}.')
            response = 'Fail'

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict
    Example of how to test this endpoint with the default lgbm model:
    *--------------EXAMPLE------------------------*
    import requests
    from ift6758.models.utils import *

    feature_list = ['shot_type', 'strength', 'is_playoff', 'prev_event_type', 'time_since_prev_event', 'is_rebound',
                    'distance_to_prev_event', 'speed_since_prev_event', 'is_penalty_shot', 'shot_distance',
                    'shot_angle', 'change_in_angle', 'time_since_pp', 'relative_strength']

    X, y, X_test, y_test = load_data(
        features=feature_list,
        train_val_seasons=DEFAULT_TRAIN_SEASONS,
        test_season=DEFAULT_TEST_SEASONS,
        do_split_val=False,
        target='is_goal',
        use_standard_scaler=True,
        drop_all_na=False,
        convert_bool_to_int=True,
        one_hot_encode_categoricals=True
    )
    X = X.drop(columns=X.columns.difference(X_test.columns))
    X_test = X_test.drop(columns=X_test.columns.difference(X.columns))
    r = requests.post("http://0.0.0.0:5000/predict", json=X_test.to_json())
    *--------------END EXAMPLE------------------------*

    Returns predictions
    """
    global CLASSIFIER
    # Get POST json data
    df_json = request.get_json()
    app.logger.info(df_json)
    df = pd.read_json(df_json)

    response = CLASSIFIER.predict_proba(df)
    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


if __name__ == '__main__':
    app.run(port=5000)