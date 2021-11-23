from comet_ml import Experiment
import os

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn import feature_extraction 
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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

def shap_feature(X, y, model):
    xgb_model = model.fit(X, y)

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X)
    
    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0])
    plt.close()
    return None


def feature_selection(df_X, df_y, model):

    # Normalization 
    # Zscore and Min-Max normalization
    cols = ['x_coordinate', 'y_coordinate', 'game_time(s)', 
            'prev_event_x', 'prev_event_y', 'time_since_prev_event',
            'distance_to_prev_event', 'speed_since_prev_event',
            'shot_distance', 'shot_angle', 'change_in_angle']
    
    df_zscore = df_X.copy()
    df_minmax = df_X.copy()
    
    for col in cols:
        # Z score normalization
        df_zscore[col] = (df_X[col] - df_X[col].mean())/df_X[col].std(ddof=0)
        # min max normalization
        df_minmax[col] = (df_X[col] - df_X[col].min())/ (df_X[col].max()- df_X[col].min())
    
    #without nomarlization
    X = df_X.to_numpy()
    y = df_y.to_numpy()
    base_scores_mean = np.mean(cross_val_score(model, X, y, cv=5))
    
    # Z-score normalization
    X = df_zscore.to_numpy()
    y = df_y.to_numpy()
    z_score_scores_mean = np.mean(cross_val_score(model, X, y, cv=5))

    # min-max normalization
    X = df_minmax.to_numpy()
    y = df_y.to_numpy()
    minmax_scores_mean = np.mean(cross_val_score(model, X, y, cv=5))

    #Removing features with low variance
    X = df_X.to_numpy()
    y = df_y.to_numpy()
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_new = sel.fit_transform(X)
    lowvar_scores_mean = np.mean(cross_val_score(model, X_new, y, cv=5))
    
    #Univariate feature selection
    database = df_minmax.to_numpy()
    X = df_minmax.to_numpy()
    y = df_y.to_numpy()
    X_new = SelectKBest(chi2, k=10).fit_transform(X, y)
    univar_scores_mean = np.mean(cross_val_score(model, X_new, y, cv=5))


    # forward search
    X = df_X.to_numpy()
    y = df_y.to_numpy()
    # start with shot_distance because it is the best relation as showing by shap figure
    X_new = X[:,11:12]
    
    scores_base = np.mean(cross_val_score(model, X_new, y, cv=5))
    index = [11]
    for i in range(X.shape[1]):
        if (i == 11):
            pass
        else:
            X_add = np.concatenate((X_new, X[:,i].reshape(-1,1)), axis=1)
            scores = np.mean(cross_val_score(model, X_add, y, cv=5))
            if (scores >= scores_base):
                X_new = X_add
                scores_base = scores
                index.append(i)
    forward_search_scores_mean = scores_base
    forward_search_idx = index

    result = {"base_scores_mean  "  :base_scores_mean,
              "z_score_scores_mean"  :z_score_scores_mean,
              "minmax_scores_mean"  :minmax_scores_mean,
              "lowvar_scores_mean"  :lowvar_scores_mean,
              "univar_scores_mean"  :univar_scores_mean,
              "forward_search_scores_mean"  :forward_search_scores_mean,
              "forward_search_idx"  :forward_search_idx}
    return result

def main():
    '''
    use the best hyperparameter
    combine with feature selection
        z-score normalization
        minmax normalization
        remove features with low variance
        Univariate feature selection
        forward search
    save result of feature selection to comet
    '''
    feature = ['period','x_coordinate','y_coordinate','is_empty_net',
                'game_time(s)','prev_event_x','prev_event_y',
                'time_since_prev_event','is_rebound','distance_to_prev_event',
                'speed_since_prev_event','shot_distance','shot_angle',
                'change_in_angle','shot_type','prev_event_type']
    
    df_X, df_y = feature_preprocessing(feature)
    
    xgb_model = XGBClassifier(
            min_child_weight =  1,
            gamma            =  0.5,
            subsample        =  0.7,
            learning_rate    =  0.01,
            colsample_bytree =  0.6,
            max_depth        =  5,
            n_estimators     =  700,
            reg_alpha        =  1.3,
            reg_lambda       =  1.1,
            objective        =  'binary:logistic',
            use_label_encoder = False,
            eval_metric      =  'error'
        )
    
    result = feature_selection(df_X, df_y, xgb_model)
    
    X = df_X.to_numpy()
    y = df_y.to_numpy()
    shap_feature(X, y, xgb_model)
    
    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="ift-6758-milestone-2",
        workspace="vanbinhtruong",
    )
    experiment.log_metrics(result)
    
    print('result is = ',result)
if __name__ == "__main__":
    main()