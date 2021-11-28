from comet_ml import Experiment

import sys
import os.path

#import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

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


DEFAULT_TRAIN_SEASONS = ['201520216', '20162017', '20172018', '20182019' ]
DEFAULT_TEST_SEASONS = ['20192020']

feature_list =['shot_type', 'strength','is_playoff', 'prev_event_type', 'time_since_prev_event', 'is_rebound', 'distance_to_prev_event','speed_since_prev_event', 'is_penalty_shot','shot_distance', 
 'shot_angle', 'change_in_angle', 'time_since_pp','relative_strength']

X_train, y_train, X_val, y_val, _, _ = utils.load_data(features = feature_list )

#clf = MLPClassifier(hidden_layer_sizes=(128,64,32,16,4), activation='relu', solver= 'adam', alpha = 0.0001, learning_rate='adaptive',learning_rate_init = 0.001, shuffle=True, random_state=8, verbose=True, max_iter=1000) #0.70
#clf = MLPClassifier(hidden_layer_sizes=(64,32,8,4), activation='relu', solver= 'adam', alpha = 0.0001, learning_rate='adaptive',learning_rate_init = 0.001, shuffle=True, random_state=8, verbose=True, max_iter=1000) 0.75
#clf = MLPClassifier(hidden_layer_sizes=(32,16,8,4), activation='relu', solver= 'adam', alpha = 0.0001, learning_rate='adaptive',learning_rate_init = 0.001, shuffle=True, random_state=8, verbose=True, max_iter=1000) 0.76


#clf = MLPClassifier(hidden_layer_sizes= (64, 32, 16, 8, 4), activation='relu', solver= 'adam', alpha = 0.004203648429726717, learning_rate='adaptive',learning_rate_init = 0.025537004473119504, shuffle=True, random_state=8, verbose=False, max_iter=1000)
#clf = MLPClassifier(hidden_layer_sizes=(128,64,32,16,8), activation='relu', solver= 'adam', alpha = 0.0001, learning_rate='adaptive',learning_rate_init = 0.0015, shuffle=True, random_state=8, verbose=True, max_iter=1000)
#clf = MLPClassifier(hidden_layer_sizes=(45), activation='relu', solver= 'adam', alpha = 0.00026682178930892837, learning_rate='adaptive',learning_rate_init = 0.001903320133403294, shuffle=True, random_state=8, verbose=True, max_iter=1000)

# clf = MLPClassifier(hidden_layer_sizes=(64), activation='relu', solver= 'adam', alpha = 0.00026682178930892837, learning_rate='adaptive',learning_rate_init = 0.001903320133403294, shuffle=True, random_state=8, verbose=True, max_iter=1000)
model = MLPClassifier(hidden_layer_sizes=(150), activation='relu', solver= 'adam', alpha =  0.0009674368057318, learning_rate='adaptive',learning_rate_init = 0.0079705459309569, shuffle=True, random_state=8, verbose=True, max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict_proba(X_val)

model_names=['Neural Network']
fig_number = '6-2'

create_figure.fig_roc_auc(y_val,predictions[:,1,None].T,fig_number,model_names,experiment=experiment)
create_figure.calibration_fig(y_val,predictions[:,1,None].T,fig_number,model_names,experiment=experiment)
create_figure.fig_cumulative_goal(y_val,predictions[:,1,None].T,fig_number,model_names,experiment=experiment)
create_figure.fig_goal_rate(y_val,predictions[:,1,None].T,fig_number,model_names,experiment=experiment)


file_name = "tuned_nn_model.pkl"

# save
pickle.dump(model, open(file_name, "wb"))

# Compute metrics
y_pred_labels = model.predict(X_val)
f1 = f1_score(y_val, y_pred_labels)
accuracy = accuracy_score(y_val, y_pred_labels)
precision = precision_score(y_val, y_pred_labels)
recall = recall_score(y_val, y_pred_labels)
metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
params = {
    "model_type": 'Neural Net',
    "scaler": "standard scaler",
    "param_grid": str(model.get_params()),
}

experiment.log_parameters(params)
experiment.log_metrics(metrics)
experiment.add_tag('6-2 NN')
experiment.log_model('6_2_nn_tuned_model', 'tuned_nn_model.pkl')

### Optuna hyperparameter search, first run done with up to 5 layers, 1 layer was presenting best result so launched second hyperparameter search only on 1 layer nn
# def objective(trial):
    
#     n_layers = 1
#     layers = []
#     for i in range(n_layers):
#         layers.append(trial.suggest_int(f'n_units_{i}', 1, 150))
    
#     alpha= trial.suggest_uniform('alpha',0.0001, 0.001)
#     lr = trial.suggest_uniform('learning_rate',0.000001, 0.05)
#     feature_list =['shot_type', 'strength','is_playoff', 'prev_event_type', 'time_since_prev_event', 'is_rebound', 'distance_to_prev_event','speed_since_prev_event', 'is_penalty_shot','shot_distance', 
#   'shot_angle', 'change_in_angle', 'time_since_pp','relative_strength']
    
#     X_train, y_train, X_val, y_val, _, _ = utils.load_data(features = feature_list )
    
#     clf = MLPClassifier(hidden_layer_sizes=layers, activation='relu', solver= 'adam', alpha = alpha, learning_rate='adaptive',learning_rate_init = lr, shuffle=True, random_state=8, verbose=False, max_iter=1000)
#     clf.fit( X_train, y_train)
#     preditions = clf.predict_proba(X_val)
#     auct=fig_roc_auc(y_val,preditions[:,1,None].T,ntries,"neural net")
#     calibration_fig(y_val,preditions[:,1,None].T,ntries,"neural net")
#     fig_goal_rate(y_val,preditions[:,1,None].T,ntries,"neural net")

#     return auct

# study = optuna.create_study(direction='maximize')
# study.optimize(objective,n_trials=100)


