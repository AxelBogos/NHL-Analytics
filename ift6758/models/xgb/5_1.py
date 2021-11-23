import os.path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibrationDisplay

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

def xgb_basic_figure(model, feature, fig_name):


    X_train, y_train, X_val, y_val, _, _ = load_data(
                                                features = feature,
                                                train_val_seasons = DEFAULT_TRAIN_SEASONS, 
                                                test_season = DEFAULT_TEST_SEASONS,
                                                train_val_ratio = 0.2, 
                                                target = 'is_goal',
                                                use_standard_scaler = False,
                                                return_as_dataframes= False,
                                                drop_all_na=  False
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict_proba(X_val)[:,1]
    y_est = [(i>=0.5)*1 for i in y_pred ]
    
    fig_roc_auc(y_val, y_pred, fig_name)
    fig_cumulative_goal(y_val, y_pred, fig_name)
    fig_goal_rate(y_val, y_pred, fig_name)
    
    return None

def main():
    '''
    create 4 figures and save and /figures with each feature
    '''
    model = XGBClassifier(use_label_encoder=False, eval_metric = 'error')
    feature = ['shot_distance']
    fig_name = '5_1_distance'
    xgb_basic_figure(model, feature, fig_name)
    
    feature = ['shot_angle']
    fig_name = '5_1_angle'
    xgb_basic_figure(model, feature, fig_name)    
    
    feature = ['shot_distance', 'shot_angle']
    fig_name = '5_1_dist_angle'
    xgb_basic_figure(model, feature, fig_name) 

if __name__ == "__main__":
    main()