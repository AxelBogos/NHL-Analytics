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

def xgb_basic_figure(model, feature):


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

    return y_val, y_pred
    

def main():
    '''
    create 4 figures and save and /figures with each feature
    '''
    y_val_vec = []
    y_pred_vec = []
    fig_name = ['5_1_dist', '5_1_angl', '5_1_dist_angl']
    
    model = XGBClassifier(use_label_encoder=False, eval_metric = 'error')
    feature = ['shot_distance']
    y_val, y_pred = xgb_basic_figure(model, feature)
    
    y_val_vec.append(y_val)
    y_pred_vec.append(y_pred)
    
    feature = ['shot_angle']
    y_val, y_pred = xgb_basic_figure(model, feature)
    
    y_val_vec.append(y_val)
    y_pred_vec.append(y_pred)  
    
    feature = ['shot_distance', 'shot_angle']
    y_val, y_pred = xgb_basic_figure(model, feature)
    
    y_val_vec.append(y_val)
    y_pred_vec.append(y_pred)
    
    fig_roc_auc(y_val_vec, y_pred_vec, fig_name)
    fig_cumulative_goal(y_val_vec, y_pred_vec, fig_name)
    fig_goal_rate(y_val_vec, y_pred_vec, fig_name)
    calibration_fig(y_val_vec, y_pred_vec, fig_name)

if __name__ == "__main__":
    main()