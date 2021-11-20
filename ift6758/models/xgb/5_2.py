from comet_ml import Experiment
import os

from sklearn.model_selection import GridSearchCV
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
from create_figure import *


def feature_preprocessing(url):
    '''
    preprocessing dataframe
    replace NaN value to 0
    encode feature
    convert dataframe to training data
    '''
    df = pd.read_csv(url)
    df_train = df.loc[(df['season'] != 20202021) & (df['season'] != 20192020)]
    
    df_feature = df_train[['period','shot_type','x_coordinate','y_coordinate','is_empty_net',
                           'game_time(s)','prev_event_type','prev_event_x','prev_event_y',
                           'time_since_prev_event','is_rebound','distance_to_prev_event',
                           'speed_since_prev_event','shot_distance','shot_angle',
                           'change_in_angle','is_goal']]
    
    
    # file NaN
    df_feature = df_feature.fillna(0)
    
    # Feature encoding for True, False
    df_feature = df_feature.replace({True: 1, False: 0})
    
    # Feature encoding for 'shot_type'
    df_feature = df_feature.replace({'shot_type':{
                                       'Backhand'   : 1,
                                       'Deflected'  : 2,
                                       'Slap Shot'  : 3,
                                       'Snap Shot'  : 4,
                                       'Tip-In'     : 5,
                                       'Wrap-around': 6,
                                       'Wrist Shot' : 7
                                    }
                                   }
                                  )
    
    # Feature encoding for 'prev_event_type'
    df_feature = df_feature.replace({'prev_event_type':{
                                      'Blocked Shot'      : 1 ,
                                      'Faceoff'           : 2 ,
                                      'Game End'          : 3 ,
                                      'Game Official'     : 4 ,
                                      'Giveaway'          : 5 ,
                                      'Goal'              : 6 ,
                                      'Hit'               : 7 ,
                                      'Missed Shot'       : 8 ,
                                      'Official Challenge': 9 ,
                                      'Penalty'           : 10,
                                      'Period End'        : 11,
                                      'Period Ready'      : 12,
                                      'Period Start'      : 13,
                                      'Shootout Complete' : 14,
                                      'Shot'              : 15,
                                      'Stoppage'          : 16,
                                      'Takeaway'          : 17
                                     }
                                    }
                                   )
    
    # convert to model data
    database = df_feature.to_numpy()
    
    X = database[:,0:-1]
    y = database[:,-1].astype(np.int32)
    
    return X, y

def xgb_grid_search(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
                                         X, y, test_size=0.2, random_state=1)
    xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic', eval_metric = 'error',
                        nthread=1)
    
    params = {
            'min_child_weight': [1],
            'gamma': [0.5],
            'subsample': [0.7],
            'learning_rate': [0.01, 0.1],
            'colsample_bytree': [0.8],
            'max_depth': [5],
            'n_estimators': [1000],
            'reg_alpha': [1.3],
            'reg_lambda': [1.1]
            }
    
    scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}
    
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=params,
        scoring = make_scorer(accuracy_score),
        n_jobs = 10,
        cv = 5,
        verbose=True
    )
    
    model = grid_search.fit(X_train, y_train)

    predict = model.predict(X_test)
    print('Best AUC Score: {}'.format(model.best_score_))
    print('Accuracy: {}'.format(accuracy_score(y_test, predict)))
    
    print(model.best_params_)
    

    return model

def main():
    url = '../../data/tidy_data.csv'
    X, y  = feature_preprocessing(url)

    model = xgb_grid_search(X, y)
    
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
        eval_metric      = 'error'
    )
    
    # create 4 figures
    y = y.reshape(-1,1)
    fig_name = '../../../figures/5_2_grid_search'
    xgboost_basic_feature(X, y, xgb_model, fig_name)
    
    # save xgb_model
    file_name = "xgb_model.pkl"
    # save
    pickle.dump(model, open(file_name, "wb"))
    
    # load
    model_loaded = pickle.load(open(file_name, "rb"))
    
    scores_mean = np.mean(cross_val_score(model_loaded, X, y, cv=5))
    
    print('score of grid search best model = ', scores_mean)
    
    os.environ["COMET_API_KEY"] = "Jt0FZk0zwp83uLiydjGVENsFg"
    
    experiment = Experiment(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="quickstart-project",
        workspace="axelbogos",
    )
    
    experiment.log_model('5_2 grid search model', 'xgb_model.pkl')

if __name__ == "__main__":
    main()

