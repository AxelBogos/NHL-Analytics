from comet_ml import Experiment
import optuna
import xgboost as xgb
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,log_loss
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from pprint import pprint
import pickle

# import figure plot
from create_figure import *

# utils_path = os.path.abspath(os.path.join('..'))
# sys.path.append(utils_path)
# from utils import *
from ift6758.models.utils import *
import os
from dotenv import load_dotenv

load_dotenv()

feature = ['period', 'x_coordinate', 'y_coordinate',
           'game_time(s)', 'prev_event_x', 'prev_event_y',
           'time_since_prev_event', 'is_rebound', 'distance_to_prev_event',
           'speed_since_prev_event', 'shot_distance', 'shot_angle',
           'change_in_angle', 'shot_type', 'prev_event_type']


def objective(trial):
    X, y, _, _ = load_data(
        features=feature,
        train_val_seasons=DEFAULT_TRAIN_SEASONS,
        test_season=DEFAULT_TEST_SEASONS,
        do_split_val=False,
        target='is_goal',
        use_standard_scaler=True,
        drop_all_na=True,
        convert_bool_to_int=True,
        one_hot_encode_categoricals=True
    )
    #X.fillna(0)


    param_grid = {
        "silent": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1, 25, 50, 75, 99, 1000]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    }

    if param_grid["booster"] == "gbtree" or param_grid["booster"] == "dart":
        param_grid["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param_grid["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param_grid["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param_grid["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param_grid["booster"] == "dart":
        param_grid["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param_grid["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param_grid["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param_grid["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    cv_scores = np.empty(3)
    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        train = xgb.DMatrix(X.iloc[train_idx], label=y[train_idx])
        val = xgb.DMatrix(X.iloc[val_idx], label=y[val_idx])

        # Add a callback for pruning.
        clf = xgb.train(param_grid, train, evals=[(val, "validation")],
                        callbacks=[XGBoostPruningCallback(trial, "validation-auc")])
        y_preds = clf.predict(val)
        score = roc_auc_score(y[val_idx], y_preds)
        cv_scores[idx] = score
    return np.mean(cv_scores)


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
        sample_train = rng.choice(X_train.shape[0], size=round(X_train.shape[0] * samp_per), replace=False)
        sample_val = rng.choice(X_val.shape[0], size=round(X_val.shape[0] * samp_per), replace=False)

        X_train = X_train[sample_train, :]
        y_train = y_train[sample_train]
        X_val = X_val[sample_val, :]
        y_val = y_val[sample_val]

    xgb = XGBClassifier(objective='binary:logistic', eval_metric='error')

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
        scoring=make_scorer(accuracy_score),
        n_jobs=10,
        cv=5,
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
    study = optuna.create_study(direction="maximize", study_name="XGB Classifier")
    func = lambda trial: objective(trial)
    study.optimize(func, n_trials=20)
    print(f"\tBest params:")

    pprint(study.best_params)
    params = study.best_params
    X_train, y_train, X_test, y_test = load_data(
        features=feature,
        train_val_seasons=DEFAULT_TRAIN_SEASONS,
        test_season=DEFAULT_TEST_SEASONS,
        do_split_val=False,
        target='is_goal',
        use_standard_scaler=True,
        drop_all_na=False,
        convert_bool_to_int=True,
        one_hot_encode_categoricals=True
    )
    # Train Model with optimal params
    experiment = Experiment(
        api_key=os.getenv('COMET_API_KEY'),
        project_name="ift-6758-milestone-2",
        workspace="axelbogos",
    )
    # model = XGBClassifier(objective='binary:logistic', **params)
    X_train = X_train.drop(columns=X_train.columns.difference(X_test.columns))
    X_test = X_test.drop(columns=X_test.columns.difference(X_train.columns))
    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]

    y_pred_vec = [y_pred]
    model_names=['Tuned XGB']
    fig_number = '5-2'
    fig_roc_auc(y_test, y_pred_vec, fig_number, model_names)
    fig_cumulative_goal(y_test, y_pred_vec, fig_number, model_names)
    fig_goal_rate(y_test, y_pred_vec, fig_number, model_names)
    calibration_fig(y_test, y_pred_vec, fig_number, model_names)

    # save xgb_model
    file_name = "tuned_xgb_model.pkl"

    # save
    pickle.dump(model, open(file_name, "wb"))

    # Compute metrics
    y_pred_labels = model.predict(X_test)
    f1 = f1_score(y_test, y_pred_labels)
    accuracy = accuracy_score(y_test, y_pred_labels)
    precision = precision_score(y_test, y_pred_labels)
    recall = recall_score(y_test, y_pred_labels)
    metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}
    params = {
        "model_type": 'XGB',
        "scaler": "standard scaler",
        "param_grid": str(model.get_params()),
    }

    experiment.log_parameters(params)
    experiment.log_metrics(metrics)
    experiment.add_tag('5-2 XGB')
    experiment.log_model('5_2 tuned model', 'tuned_xgb_model.pkl')


if __name__ == "__main__":
    main()
