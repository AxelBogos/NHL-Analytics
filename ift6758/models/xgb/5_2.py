from comet_ml import Experiment

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import make_scorer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

# import figure plot
from create_figure import *

# import utils
import sys
utils_path = os.path.abspath(os.path.join('..'))
sys.path.append(utils_path)
from utils import *
import os
from dotenv import load_dotenv
load_dotenv()
COMET_API_KEY = os.getenv('COMET_API_KEY') 

def xgb_grid_search(X_train, X_val, y_train, y_val, samp_per):
    '''
    Grid search for the best hyper parameter
    because database too large, random sampling x% of train, val for searching param
        0.1: random sampling 10% of train, val
        1  : do not sampling database
    ''' 
    rng = np.random.default_rng()
    # random sampling
    if samp_per < 1:
        sample_train = rng.choice(X_train.shape[0], size = round(X_train.shape[0]*samp_per), replace=False)
        sample_val   = rng.choice(X_val.shape[0], size = round(X_val.shape[0]*samp_per), replace=False)
        
        X_train = X_train[sample_train, :]
        y_train = y_train[sample_train]
        X_val = X_val[sample_val, :]
        y_val = y_val[sample_val]

    
    xgb = XGBClassifier(objective='binary:logistic', eval_metric = 'error')
    
    params = {
            'min_child_weight': [1],
            'gamma': [0.5],
            'subsample': [0.7],
            'learning_rate': [0.01, 0.1, 0.2],
            'colsample_bytree': [0.4, 0.6, 0.8],
            'max_depth': [4, 5, 6],
            'n_estimators': [400, 700, 1000],
            'reg_alpha': [1.3],
            'reg_lambda': [1.1]
            }
    
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=params,
        scoring = make_scorer(accuracy_score),
        n_jobs = 10,
        cv = 5,
        verbose=True
    )
    
    model = grid_search.fit(X_train, y_train)

    predict = model.predict(X_val)
    accuracy = accuracy_score(y_val, predict)
    print('Best AUC Score: {}'.format(model.best_score_))
    print('Accuracy: {}'.format(accuracy))
    
    print(model.best_params_)
    

    return model, accuracy

def main():
    '''
    grid search for the best model
    save figures for the best parameter
    save model, hyperparameter
    '''
    feature = ['period','x_coordinate','y_coordinate',
                'game_time(s)','prev_event_x','prev_event_y',
                'time_since_prev_event','is_rebound','distance_to_prev_event',
                'speed_since_prev_event','shot_distance','shot_angle',
                'change_in_angle','shot_type','prev_event_type']
    df_X, df_y = feature_preprocessing(feature)
    
    X = df_X.to_numpy()
    y = (df_y.to_numpy()).astype(int)
    
    # split Train, valid
    X_train, X_val, y_train, y_val = train_test_split(
                                         X, y, test_size=0.2, random_state=1)
    
    
    model, accuracy = xgb_grid_search(X_train, X_val, y_train, y_val, 1)
    
    xgb_model = XGBClassifier(
        colsample_bytree =  model.best_params_['colsample_bytree'],
        gamma            =  model.best_params_['gamma'],
        learning_rate    =  model.best_params_['learning_rate'],
        max_depth        =  model.best_params_['max_depth'],
        min_child_weight =  model.best_params_['min_child_weight'],
        n_estimators     =  model.best_params_['n_estimators'],
        reg_alpha        =  model.best_params_['reg_alpha'],
        reg_lambda       =  model.best_params_['reg_lambda'],
        subsample        =  model.best_params_['subsample'],
        use_label_encoder=  False,
        objective        =  'binary:logistic',
        eval_metric      =  'error'
    )
    
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict_proba(X_val)[:,1]
    y_est = [(i>=0.5)*1 for i in y_pred ]
    
    fig_name = '5_2_grid_search'
    fig_roc_auc(y_val, y_pred, fig_name)
    fig_cumulative_goal(y_val, y_pred, fig_name)
    fig_goal_rate(y_val, y_pred, fig_name)
    calibration_fig(y_val, y_pred, fig_name)
    
    # save xgb_model
    file_name = "xgb_model.pkl"
    # save
    pickle.dump(model, open(file_name, "wb"))
    
    # load
    model_loaded = pickle.load(open(file_name, "rb"))

    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="ift-6758-milestone-2",
        workspace="vanbinhtruong",
    )
    experiment.log_parameters(xgb_model.get_params())
    experiment.log_metric("accuracy", accuracy)
    experiment.log_model('5_2 grid search model', 'xgb_model.pkl')

if __name__ == "__main__":
    main()
