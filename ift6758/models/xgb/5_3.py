from comet_ml import Experiment
import os

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from create_figure import *


def feature_preprocessing(url):
    '''
    preprocessing dataframe
    replace NaN value to 0
    encode feature
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
       
    return df_feature

def shap_feature(X, y, model):
    xgb_model = model.fit(X, y)

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X)
    
    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0])
    
    return None


def feature_selection(df, model):

    # Normalization 
    # Zscore and Min-Max normalization
    cols = list(df_feature.columns)
    cols.remove('period')
    cols.remove('shot_type')
    cols.remove('is_empty_net')
    cols.remove('prev_event_type')
    cols.remove('is_rebound')
    cols.remove('is_goal')
    
    df_zscore = df_feature.copy()
    df_minmax = df_feature.copy()
    
    
    for col in cols:
        # Z score normalization
        df_zscore[col] = (df_feature[col] - df_feature[col].mean())/df_feature[col].std(ddof=0)
        # min max normalization
        df_minmax[col] = (df_feature[col] - df_feature[col].min())/ (df_feature[col].max()- df_feature[col].min())
    
    #without nomarlization
    database = df_feature.to_numpy()
    X = database[:,0:-1]
    y = database[:,-1].astype(np.int32)
    base_scores_mean = np.mean(cross_val_score(model, X, y, cv=5))
    
    # Z-score normalization
    database = df_zscore.to_numpy()
    X = database[:,0:-1]
    y = database[:,-1].astype(np.int32)
    z_score_scores_mean = np.mean(cross_val_score(model, X, y, cv=5))

    # min-max normalization
    database = df_minmax.to_numpy()
    X = database[:,0:-1]
    y = database[:,-1].astype(np.int32)
    minmax_scores_mean = np.mean(cross_val_score(model, X, y, cv=5))

    #Removing features with low variance
    database = df_feature.to_numpy()
    X = database[:,0:-1]
    y = database[:,-1].astype(np.int32)
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X_new = sel.fit_transform(X)
    lowvar_scores_mean = np.mean(cross_val_score(model, X_new, y, cv=5))
    
    #Univariate feature selection
    database = df_minmax.to_numpy()
    X = database[:,0:-1]
    y = database[:,-1].astype(np.int32)
    X_new = SelectKBest(chi2, k=10).fit_transform(X, y)
    univar_scores_mean = np.mean(cross_val_score(model, X_new, y, cv=5))


    # forward search
    # start with shot_distance because it is the best relation as showing by shap figure
    X_new = X[:,13:14]
    
    scores_base = np.mean(cross_val_score(model, X_new, y, cv=5))
    index = [13]
    for i in range(X.shape[1]):
        if (i == 13):
            pass
        else:
            X_add = np.concatenate((X_new, X[:,i].reshape(-1,1)), axis=1)
            scores = np.mean(cross_val_score(model, X_add, y, cv=5))
            if (scores >= scores_base):
                X_new = X_add
                scores_base = scores
                index.append(i)
    forward_search_scores_mean = scores_mean
    forward_search_idx = index

    return base_scores_mean, z_score_scores_mean, minmax_scores_mean, lowvar_scores_mean, univar_scores_mean, forward_search_scores_mean, forward_search_idx

def main():
    url = '../../data/tidy_data.csv'
    xgb_model = XGBClassifier(
            min_child_weight =  1,
            gamma            =  0.5,
            subsample        =  0.7,
            learning_rate    =  0.01,
            colsample_bytree =  0.8,
            max_depth        =  5,
            n_estimators     =  1000,
            reg_alpha        =  1.3,
            reg_lambda       =  1.1,
            objective        =  'binary:logistic',
            eval_metric      = 'error'
        )
    df_feature = feature_preprocessing(url)

    base_scores_mean, z_score_scores_mean, minmax_scores_mean, lowvar_scores_mean, \
        univar_scores_mean, forward_search_scores_mean, forward_search_idx = feature_selection(df_feature, xgb_model)

    database = df_feature.to_numpy()
    X = database[:,0:-1]
    y = database[:,-1].astype(np.int32)
    shap_feature(X, y, xgb_model)

if __name__ == "__main__":
    main()