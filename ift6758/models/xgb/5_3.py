import shap
from comet_ml import Experiment
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import chi2
from xgboost import XGBClassifier
# import figure plot
from sklearn.model_selection import cross_val_score

from create_figure import *

# utils_path = os.path.abspath(os.path.join('..'))
# sys.path.append(utils_path)
# from utils import *
from ift6758.models.utils import *
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
    feature = ['period', 'x_coordinate', 'y_coordinate',
               'game_time(s)', 'prev_event_x', 'prev_event_y',
               'time_since_prev_event', 'is_rebound', 'distance_to_prev_event',
               'speed_since_prev_event', 'shot_distance', 'shot_angle',
               'change_in_angle', 'shot_type', 'prev_event_type', 'time_since_pp',
               'home_strength', 'away_strength']
    
    df_X, df_y,_,_ = load_data(feature, do_split_val=False, one_hot_encode_categoricals=True)
    params = {'objective': 'binary:logistic', 'use_label_encoder': True, 'base_score': 0.5, 'booster': 'dart', 'colsample_bylevel': 1,
              'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 1.0644957574704856e-06, 'gpu_id': -1, 'importance_type': 'gain',
              'interaction_constraints': '', 'learning_rate': 0.253149986, 'max_delta_step': 0, 'max_depth': 8, 'min_child_weight': 1,
              'monotone_constraints': '()', 'n_estimators': 709, 'n_jobs': 4, 'num_parallel_tree': 1, 'random_state': 0,
              'reg_alpha': 1.86814141e-05, 'reg_lambda': 1.74496799e-05, 'scale_pos_weight': 4, 'subsample': 1,
              'tree_method': 'exact','validate_parameters': 1, 'verbosity': None, 'alpha': 1.8681414898850136e-05,
              'eta': 0.2531499873091687,'grow_policy': 'depthwise', 'lambda': 1.744968010669035e-05,
              'normalize_type': 'forest', 'rate_drop': 0.047822250564598025,'sample_type': 'uniform',
              'skip_drop': 0.00026092912897969413}
    xgb_model = XGBClassifier(**params)
    
    result = feature_selection(df_X, df_y, xgb_model)
    
    X = df_X.to_numpy()
    y = df_y.to_numpy()
    shap_feature(X, y, xgb_model)
    
    experiment = Experiment(
        api_key=COMET_API_KEY,
        project_name="ift-6758-milestone-2",
        workspace="axelbogos",
    )
    experiment.log_metrics(result)
    #TODO LOG MODEL & metrics
    
    print('result is = ',result)
if __name__ == "__main__":
    main()