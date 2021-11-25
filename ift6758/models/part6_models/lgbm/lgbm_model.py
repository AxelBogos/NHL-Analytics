from comet_ml import Experiment
import os
from typing import List
from pprint import pprint
import lightgbm as lgbm
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from ift6758.models.utils import load_data

load_dotenv()

RANDOM_SEED = 1237342
FEATURES = ['shot_type', 'is_rebound', 'distance_to_prev_event', 'is_penalty_shot', 'shot_distance', 'shot_angle',
            'change_in_angle']


def run_experiment(features: List[str], use_comet=False, params=None, target='is_goal'):
    # Comet set-up
    if use_comet:
        exp = Experiment(
            project_name="ift-6758-milestone-2",
            workspace="axelbogos",
            auto_output_logging="default",
            api_key=os.environ.get("COMET_API"),
        )

    # Load data
    train, test = load_data(features=features, target=target)
    CATEGORICAL_COLUMNS = train.select_dtypes(exclude=["number", "bool_"]).columns

    # drop NAs
    train = train.dropna()
    test = test.dropna()

    # Convert bools to ints
    train[train.select_dtypes([bool]).columns] = train.select_dtypes([bool]).astype(int)
    test[test.select_dtypes([bool]).columns] = test.select_dtypes([bool]).astype(int)

    # X, y split
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]

    # Scale non-categorical features
    scaler = StandardScaler()
    X_train[X_train.columns.difference(CATEGORICAL_COLUMNS)] = scaler.fit_transform(
        X_train[X_train.columns.difference(CATEGORICAL_COLUMNS)])
    X_test[X_test.columns.difference(CATEGORICAL_COLUMNS)] = scaler.fit_transform(
        X_test[X_test.columns.difference(CATEGORICAL_COLUMNS)])

    # one-hot categorical columns
    X_train = pd.get_dummies(data=X_train, columns=CATEGORICAL_COLUMNS)
    X_test = pd.get_dummies(data=X_test, columns=CATEGORICAL_COLUMNS)

    # Fit/Predict
    if params:
        clf = lgbm.LGBMClassifier(**params)
    else:
        clf = lgbm.LGBMClassifier()
        params = clf.get_params()  # default hyper-params

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Record Metrics
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
    pprint(metrics)
    # Add to params
    comet_params = {
        "random_state": RANDOM_SEED,
        "model_type": 'LGBM',
        "scaler": "Standard Scaler",
        "param_grid": str(params),
    }

    # Log
    if use_comet:
        exp.log_dataset_hash(X_train)
        exp.log_parameters(comet_params)
        exp.log_metrics(metrics)
        exp.add_tag('LGBM')

params = {'n_estimators': 10000,
          'learning_rate': 0.02443469298850732,
          'num_leaves': 1460,
          'max_depth': 12,
          'min_data_in_leaf': 7100,
          'lambda_l1': 35,
          'lambda_l2': 20,
          'min_gain_to_split': 2.6845705487349596,
          'bagging_fraction': 0.8,
          'bagging_freq': 1,
          'feature_fraction': 0.5}
run_experiment(FEATURES, params=params, use_comet=False)
