from comet_ml import Experiment

import sys
import os.path

import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
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


DEFAULT_TRAIN_SEASONS = ['20152016', '20162017', '20172018', '20182019' ]
DEFAULT_TEST_SEASONS = ['20192020']

feature_list =['shot_type', 'strength','is_playoff', 'prev_event_type', 'time_since_prev_event', 'is_rebound', 'distance_to_prev_event','speed_since_prev_event', 'is_penalty_shot','shot_distance', 
 'shot_angle', 'change_in_angle', 'time_since_pp','relative_strength']

X_train, y_train, X_val, y_val, _, _ = utils.load_data(features = feature_list,drop_all_na=True, one_hot_encode_categoricals = True )

model = MLPClassifier(hidden_layer_sizes=(138), activation='relu', solver= 'adam', alpha =0.00036981301414230866,  learning_rate='adaptive',learning_rate_init = 0.0063417839151296606, shuffle=True, random_state=8, verbose=True, max_iter=1000)
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
#         layers.append(trial.suggest_int(f'n_units_{i}', 120, 180))
    
#     alpha= trial.suggest_uniform('alpha',0.0001, 0.002)
#     lr = trial.suggest_uniform('learning_rate',0.0001, 0.05)
#     feature_list =['shot_type', 'strength','is_playoff', 'prev_event_type', 'time_since_prev_event', 'is_rebound', 'distance_to_prev_event','speed_since_prev_event', 'is_penalty_shot','shot_distance', 
#   'shot_angle', 'change_in_angle', 'time_since_pp','relative_strength']
    
#     X_train, y_train, X_val, y_val, _, _ = utils.load_data(features = feature_list )
    
#     clf = MLPClassifier(hidden_layer_sizes=layers, activation='relu', solver= 'adam', alpha = alpha, learning_rate='adaptive',learning_rate_init = lr, shuffle=True, random_state=8, verbose=False, max_iter=1000)
#     clf.fit( X_train, y_train)
#     preditions = clf.predict_proba(X_val)
#     create_figure.fig_roc_auc(y_val,preditions[:,1,None].T,0,"neural net")
#     create_figure.calibration_fig(y_val,preditions[:,1,None].T,0,"neural net")
#     create_figure.fig_goal_rate(y_val,preditions[:,1,None].T,0,"neural net")
    
#     auc = roc_auc_score(y_val, preditions[:,1,None])
    
#     return auc

# study = optuna.create_study(direction='maximize')
# study.optimize(objective,n_trials=100)


# Trial 30 finished with value: 0.7687268066245888 and parameters: {'n_units_0': 155, 'alpha': 0.0017677820327632425, 'learning_rate': 0.005797883282249508}. Best is trial 2 with value: 0.7719897469955013.
# Trial 47 finished with value: 0.7699378891564603 and parameters: {'n_units_0': 120, 'alpha': 0.0004473670754349809, 'learning_rate': 0.01236461944647371}. Best is trial 2 with value: 0.7719897469955013.
# Trial 51 finished with value: 0.7712301993003329 and parameters: {'n_units_0': 138, 'alpha': 0.0003698130141423086, 'learning_rate': 0.0063417839151296606}. Best is trial 2 with value: 0.7719897469955013.
# Trial 58 finished with value: 0.7701654286262516 and parameters: {'n_units_0': 167, 'alpha': 0.00019129456040594076, 'learning_rate': 0.002156743366986701}. Best is trial 2 with value: 0.7719897469955013.