from comet_ml import Experiment
from create_figure import *
from ift6758.models import utils
from q3_2 import *
from q3_1 import *
from q3_3 import *


if __name__ == "__main__":

    roc_data = []
    tidy = pd.read_csv("../../data/tidy_data.csv", sep=",")
    feature_list = ['shot_distance']
    X_train, y_train, X_val, y_val, _, _ = utils.load_data(features=feature_list)
    Q1_train_regression_evaluate(X_train, y_train, X_val, y_val)
    model_r, y_valid_r, predictions_r_prob, y_prediction_r = Q2_regression_random(X_train, y_train, X_val, y_val)
    model, y_valid, predictions_prob, y_prediction = Q2_regression_distance(X_train, y_train, X_val, y_val)

    fpr_d, tpr_d, _ = metrics.roc_curve(y_valid, predictions_prob[::, 1])
    auc_distance = metrics.roc_auc_score(y_valid, predictions_prob[::, 1])
    roc_data.append([fpr_d, tpr_d, auc_distance, 'distance'])

    fpr_r, tpr_r, _ = metrics.roc_curve(y_valid_r, predictions_r_prob)
    auc_random = metrics.roc_auc_score(y_valid_r, predictions_r_prob)
    roc_data.append([fpr_r, tpr_r, auc_random, 'random'])

    feature_list = [ 'shot_angle']
    X_train, y_train, X_val, y_val, _, _ = utils.load_data(features=feature_list)
    model_angle, y_valid_angle, predictions_prob_angle, y_prediction_angle = Q3_regre_angle(X_train, y_train, X_val, y_val)

    feature_list = ['shot_distance', 'shot_angle']
    X_train, y_train, X_val, y_val, _, _ = utils.load_data(features=feature_list)
    model_angle_dist, y_valid_angle_dist, predictions_prob_angle_dist, y_prediction_angle_dist = Q3_regression_angle_distance(X_train, y_train, X_val, y_val)

    fpr_a, tpr_a, _ = metrics.roc_curve(y_valid_angle, predictions_prob_angle[::, 1])
    auc_angle = metrics.roc_auc_score(y_valid_angle, predictions_prob_angle[::, 1])
    roc_data.append([fpr_a, tpr_a, auc_angle, 'angle'])

    fpr_a_d, tpr_a_d, _ = metrics.roc_curve(y_valid_angle_dist, predictions_prob_angle_dist[::, 1])
    auc_angle_d = metrics.roc_auc_score(y_valid_angle_dist, predictions_prob_angle_dist[::, 1])
    roc_data.append([fpr_a_d, tpr_a_d, auc_angle_d, 'angle_distance'])

    plot_roc_curve(roc_data, 'AUC_data')

    x_r, y_r = goal_rate_data(np.squeeze(y_valid_r), predictions_r_prob)
    x_a, y_a = goal_rate_data(np.squeeze(y_valid_angle), np.squeeze(predictions_prob_angle[:, 1]))
    x_d, y_d = goal_rate_data(np.squeeze(y_valid), np.squeeze(predictions_prob[:, 1]))
    x_d_a, y_d_a = goal_rate_data(np.squeeze(y_valid_angle_dist), np.squeeze(predictions_prob_angle_dist[:, 1]))
    list_goal_rate=[]
    list_goal_rate.append([x_r, y_r, 'random'])
    list_goal_rate.append([x_a, y_a, 'angle'])
    list_goal_rate.append([x_d, y_d, 'distance'])
    list_goal_rate.append([x_d_a, y_d_a, 'distance-angle'])
    fig_goal_rate(list_goal_rate)


    df_r = cumulative_goal_data(np.squeeze(y_valid_r), predictions_r_prob)
    df_a = cumulative_goal_data(np.squeeze(y_valid_angle), np.squeeze(predictions_prob_angle[:, 1]))
    df_d = cumulative_goal_data(np.squeeze(y_valid), np.squeeze(predictions_prob[:, 1]))
    df_d_a = cumulative_goal_data(np.squeeze(y_valid_angle_dist), np.squeeze(predictions_prob_angle_dist[:, 1]))
    list_cumul_data=[]
    df_r.cat = 'Random'
    df_a.cat = 'Angle'
    df_d.cat = 'Distance'
    df_d_a.cat = 'Distance-Angle'
    list_cumul_data.append(df_r)
    list_cumul_data.append(df_a)
    list_cumul_data.append(df_d)
    list_cumul_data.append(df_d_a)
    fig_cumulative_goal(list_cumul_data)

    list_calibration_data=[]

    list_calibration_data.append([y_valid, predictions_prob[:, 1], 'Distance'])
    list_calibration_data.append([y_valid_r, predictions_r_prob, 'Random'])
    list_calibration_data.append([y_valid_angle, predictions_prob_angle[:, 1], 'Angle'])
    list_calibration_data.append([y_valid_angle_dist, predictions_prob_angle_dist[:, 1], 'Distance-Angle'])

    calibration_plot(list_calibration_data)
