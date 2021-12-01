from comet_ml import API
from comet_ml import Experiment

import os, sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import StackingClassifier


import pickle
import lightgbm as lgbm
from xgboost import XGBClassifier

import pandas as pd
import numpy as np

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import utils
import create_figure

COMET_API_KEY = os.getenv('COMET_API_KEY')

experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="ift-6758-milestone-2",
        workspace="axelbogos",

    )




### LOADING MODELS FROM COMET REGISTRY
api = API(api_key=COMET_API_KEY)

api.download_registry_model('axelbogos', '6-lgbm', version='1.0.0')
api.download_registry_model('axelbogos', '6-2-nn-tuned-model', version='1.0.0')
api.download_registry_model('axelbogos', '6-3-adaboost-tuned-model', version='1.0.0')
api.download_registry_model('axelbogos', '6-4-stacked-trained-tuned-model', version='1.0.0')
api.download_registry_model('axelbogos', '5-2-grid-search-model', version='2.0.0')
api.download_registry_model('axelbogos', '3-3-angle-dist-logreg-model', version='1.0.0')
api.download_registry_model('axelbogos', '3-3-angle-logreg-model', version='1.0.0')
api.download_registry_model('axelbogos', '3-1-dist-logreg-model', version='1.0.0')

lgbm_model = pickle.load(open('6-LGBM.pkl','rb'))
nn = pickle.load(open('tuned_nn_model.pkl','rb'))
adaboost = pickle.load(open('tuned_adaboost_model.pkl','rb'))
stack_train = pickle.load(open('tuned_stacked_trained_model.pkl','rb'))
xgb = pickle.load(open('tuned_xgb_model.pkl','rb'))
logreg_dist_angle = pickle.load(open('LogReg_dist_angle_model.pkl','rb'))
logreg_dist = pickle.load(open('LogReg_dist_model.pkl','rb'))
logreg_angle = pickle.load(open('LogReg_angle_model.pkl','rb'))



### LOADING DATA
feature_list = ['period', 'x_coordinate', 'y_coordinate',
           'game_time(s)', 'prev_event_x', 'prev_event_y',
           'time_since_prev_event', 'is_rebound', 'distance_to_prev_event',
           'speed_since_prev_event', 'shot_distance', 'shot_angle',
           'change_in_angle', 'shot_type', 'prev_event_type','time_since_pp',
           'home_strength','away_strength', 'strength', 'relative_strength','is_penalty_shot', 'shot_type', 'is_playoff']

X_train, y_train, X_val, y_val, X_test, y_test = utils.load_data(features = feature_list,drop_all_na=True, one_hot_encode_categoricals = True )

#Refitting everything to make sure there aren't any issue
lgbm_model.fit(X_train, y_train)
adaboost.fit(X_train,y_train)
nn.fit(X_train,y_train)
stack_train.fit(X_train,y_train)
xgb.fit(X_train,y_train)




feature_list_nn =['strength', 'is_playoff', 'time_since_prev_event', 'is_rebound',
       'distance_to_prev_event', 'speed_since_prev_event',
       'is_penalty_shot', 'shot_distance', 'shot_angle',
       'change_in_angle', 'time_since_pp', 'relative_strength',
       'shot_type_Backhand', 'shot_type_Deflected', 'shot_type_Slap Shot',
       'shot_type_Snap Shot', 'shot_type_Tip-In', 'shot_type_Wrap-around',
       'shot_type_Wrist Shot', 'prev_event_type_Blocked Shot',
       'prev_event_type_Faceoff', 'prev_event_type_Giveaway',
       'prev_event_type_Goal', 'prev_event_type_Hit',
       'prev_event_type_Missed Shot', 'prev_event_type_Penalty',
       'prev_event_type_Shot', 'prev_event_type_Takeaway']

feature_list_adaboost =['strength', 'is_playoff', 'time_since_prev_event', 'is_rebound',
       'distance_to_prev_event', 'speed_since_prev_event',
       'is_penalty_shot', 'shot_distance', 'shot_angle',
       'change_in_angle', 'time_since_pp', 'relative_strength',
       'shot_type_Backhand', 'shot_type_Deflected', 'shot_type_Slap Shot',
       'shot_type_Snap Shot', 'shot_type_Tip-In', 'shot_type_Wrap-around',
       'shot_type_Wrist Shot', 'prev_event_type_Blocked Shot',
       'prev_event_type_Faceoff', 'prev_event_type_Giveaway',
       'prev_event_type_Goal', 'prev_event_type_Hit',
       'prev_event_type_Missed Shot', 'prev_event_type_Penalty',
       'prev_event_type_Shot', 'prev_event_type_Takeaway']

feature_list_stack_trained = ['period', 'x_coordinate', 'y_coordinate', 'strength',
       'game_time(s)', 'prev_event_x', 'prev_event_y',
       'time_since_prev_event', 'is_rebound', 'distance_to_prev_event',
       'speed_since_prev_event', 'shot_distance', 'shot_angle',
       'change_in_angle', 'home_strength', 'away_strength',
       'time_since_pp', 'relative_strength', 'shot_type_Backhand',
       'shot_type_Deflected', 'shot_type_Slap Shot',
       'shot_type_Snap Shot', 'shot_type_Tip-In', 'shot_type_Wrap-around',
       'shot_type_Wrist Shot', 'prev_event_type_Blocked Shot',
       'prev_event_type_Faceoff', 'prev_event_type_Giveaway',
       'prev_event_type_Goal', 'prev_event_type_Hit',
       'prev_event_type_Missed Shot', 'prev_event_type_Penalty',
       'prev_event_type_Shot', 'prev_event_type_Takeaway']


predictions_lgbm = lgbm.predict_proba(X_val)
predictions_nn = nn.predict_proba(X_val[feature_list_nn])
predictions_adaboost = adaboost.predict_proba(X_val[feature_list_adaboost])
predictions_stack_train = stack_train.predict_proba(X_val[feature_list_stack_trained])





data_graph = [predictions_lgbm[:,1,None],predictions_nn[:,1,None], predictions_adaboost[:,1,None],predictions_stack_train[:,1,None]]

 
model_names = ['Light Gradient Boosting Machine', 'Neural Network', 'AdaBoost', 'Stacked Classifiers']
fig_number = '6'
 
#creating figures and saving them locally + on comet
create_figure.fig_roc_auc(y_val, data_graph,fig_number,model_names=model_names,experiment=experiment)
create_figure.calibration_fig(y_val, data_graph,fig_number,model_names=model_names,experiment=experiment)
create_figure.fig_cumulative_goal(y_val, data_graph,fig_number,model_names=model_names,experiment=experiment)
create_figure.fig_goal_rate(y_val, data_graph,fig_number,model_names=model_names,experiment=experiment, rounds=5)

experiment.add_tag('6 Combined plot')



### plots Q7.1
predictions_stack_train = stack_train.predict_proba(X_test[X_test.is_playoff== 0][feature_list_stack_trained])
predictions_logreg_distance_angle = logreg_dist_angle.predict_proba(X_test[X_test.is_playoff== 0][['shot_distance','shot_angle']])
predictions_logreg_distance = logreg_dist.predict_proba(X_test[X_test.is_playoff== 0][['shot_distance']])
predictions_logreg_angle = logreg_angle.predict_proba(X_test[X_test.is_playoff== 0][['shot_angle']])
predictions_xgb = xgb.predict_proba(X_test[X_test.is_playoff== 0])


data_graph = [predictions_stack_train[:,1,None],predictions_logreg_distance_angle[:,1,None], predictions_logreg_distance[:,1,None],predictions_logreg_angle[:,1,None],predictions_xgb[:,1,None]]

 
model_names = ['Stacked Classifiers', 'Log Regression Distance Angle', 'Log Regression Distance', 'Log Regression Angle','XGBoost']
fig_number = '2019-2020 Regular Season'
 
#creating figures and saving them locally + on comet
create_figure.fig_roc_auc(y_test[X_test.is_playoff== 0], data_graph,fig_number,model_names=model_names,experiment=experiment)
create_figure.calibration_fig(y_test[X_test.is_playoff== 0], data_graph,fig_number,model_names=model_names,experiment=experiment)
create_figure.fig_cumulative_goal(y_test[X_test.is_playoff== 0], data_graph,fig_number,model_names=model_names,experiment=experiment)
create_figure.fig_goal_rate(y_test[X_test.is_playoff== 0], data_graph,fig_number,model_names=model_names,experiment=experiment, rounds=5)


### plots Q7.2
predictions_stack_train = stack_train.predict_proba(X_test[X_test.is_playoff== 1][feature_list_stack_trained])
predictions_logreg_distance_angle = logreg_dist_angle.predict_proba(X_test[X_test.is_playoff== 1][['shot_distance','shot_angle']])
predictions_logreg_distance = logreg_dist.predict_proba(X_test[X_test.is_playoff== 1][['shot_distance']])
predictions_logreg_angle = logreg_angle.predict_proba(X_test[X_test.is_playoff== 1][['shot_angle']])
predictions_xgb = xgb.predict_proba(X_test[X_test.is_playoff== 1])


data_graph = [predictions_stack_train[:,1,None],predictions_logreg_distance_angle[:,1,None], predictions_logreg_distance[:,1,None],predictions_logreg_angle[:,1,None],predictions_xgb[:,1,None]]

 
model_names = ['Stacked Classifiers', 'Log Regression Distance Angle', 'Log Regression Distance', 'Log Regression Angle','XGBoost']
fig_number = '2019-2020 Playoff'
 
#creating figures and saving them locally + on comet
create_figure.fig_roc_auc(y_test[X_test.is_playoff==1], data_graph,fig_number,model_names=model_names,experiment=experiment)
create_figure.calibration_fig(y_test[X_test.is_playoff== 1], data_graph,fig_number,model_names=model_names,experiment=experiment)
create_figure.fig_cumulative_goal(y_test[X_test.is_playoff== 1], data_graph,fig_number,model_names=model_names,experiment=experiment)
create_figure.fig_goal_rate(y_test[X_test.is_playoff== 1], data_graph,fig_number,model_names=model_names,experiment=experiment, rounds=5)
experiment.add_tag('7 Combined plot')
    

