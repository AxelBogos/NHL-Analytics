from comet_ml import Experiment
from create_figure import *
from q3_2 import *
from q3_1 import *
from q3_3 import *


if __name__ == "__main__":
    # Create an experiment with your api key


    roc_data = []
    tidy = pd.read_csv("../../data/tidy_data.csv", sep=",")
    tidy = tidy.loc[:, ("shot_distance", "is_empty_net", "shot_angle", "is_goal")]
    tidy = tidy.replace({False: 0, True: 1})
    tidy["is_empty_net"] = tidy["is_empty_net"].fillna(-1)
    tidy = tidy.dropna()
    train_regression_evaluate(tidy)
    model_r, x_test_r, y_test_r, predictions_r_prob = regression_random(tidy)
    model, x_test, y_test, predictions_prob = regression_distance(tidy)

    fpr_d, tpr_d, _ = metrics.roc_curve(y_test, predictions_prob[::, 1])
    auc_distance = metrics.roc_auc_score(y_test, predictions_prob[::, 1])
    roc_data.append([fpr_d, tpr_d, auc_distance, 'distance'])

    fpr_r, tpr_r, _ = metrics.roc_curve(y_test_r, predictions_r_prob[::, 1])
    auc_random = metrics.roc_auc_score(y_test_r, predictions_r_prob[::, 1])
    roc_data.append([fpr_r, tpr_r, auc_random, 'random'])

    model_r_a, x_test_r_a, y_test_r_a, predictions_r_prob_a = regression_angle_random(tidy)
    model_angle, x_test_angle, y_test_angle, predictions_prob_angle = regre_angle(tidy)
    model_angle_dist, x_test_angle_dist, y_test_angle_dist, predictions_prob_angle_dist = regression_angle_distance(tidy)

    fpr_a, tpr_a, _ = metrics.roc_curve(y_test_angle, predictions_prob_angle[::, 1])
    auc_angle = metrics.roc_auc_score(y_test_angle, predictions_prob_angle[::, 1])
    roc_data.append([fpr_a, tpr_a, auc_angle, 'angle'])

    fpr_a_d, tpr_a_d, _ = metrics.roc_curve(y_test_angle_dist, predictions_prob_angle_dist[::, 1])
    auc_angle_d = metrics.roc_auc_score(y_test_angle_dist, predictions_prob_angle_dist[::, 1])
    roc_data.append([fpr_a_d, tpr_a_d, auc_angle_d, 'angle_distance'])

    plot_roc_curve(roc_data, 'AUC_data')

    x_r, y_r = goal_rate_data(np.squeeze(y_test_r), np.squeeze(predictions_r_prob[:, 1]))
    x_a, y_a = goal_rate_data(np.squeeze(y_test_angle), np.squeeze(predictions_prob_angle[:, 1]))
    x_d, y_d = goal_rate_data(np.squeeze(y_test), np.squeeze(predictions_prob[:, 1]))
    x_d_a, y_d_a = goal_rate_data(np.squeeze(y_test_angle_dist), np.squeeze(predictions_prob_angle_dist[:, 1]))
    list_goal_rate=[]
    list_goal_rate.append([x_r, y_r, 'random'])
    list_goal_rate.append([x_a, y_a, 'angle'])
    list_goal_rate.append([x_d, y_d, 'distance'])
    list_goal_rate.append([x_d_a, y_d_a, 'distance-angle'])
    fig_goal_rate(list_goal_rate)


    df_r = cumulative_goal_data(np.squeeze(y_test_r), np.squeeze(predictions_r_prob[:, 1]))
    df_a = cumulative_goal_data(np.squeeze(y_test_angle), np.squeeze(predictions_prob_angle[:, 1]))
    df_d = cumulative_goal_data(np.squeeze(y_test), np.squeeze(predictions_prob[:, 1]))
    df_d_a = cumulative_goal_data(np.squeeze(y_test_angle_dist), np.squeeze(predictions_prob_angle_dist[:, 1]))
    list_cumul_data=[]
    df_r.cat = 'random'
    df_a.cat = 'angle'
    df_d.cat = 'distance'
    df_d_a.cat = 'distance-angle'
    list_cumul_data.append(df_r)
    list_cumul_data.append(df_a)
    list_cumul_data.append(df_d)
    list_cumul_data.append(df_d_a)
    fig_cumulative_goal(list_cumul_data)

    list_calibration_data=[]
    list_calibration_data.append([x_test, y_test, model, 'Distance'])
    list_calibration_data.append([x_test_r, y_test_r, model_r, 'Random'])
    list_calibration_data.append([x_test_angle, y_test_angle, model_angle, 'Angle'])
    list_calibration_data.append([x_test_angle_dist, y_test_angle_dist, model_angle_dist, 'Distance-Angle'])

    calibration_plot(list_calibration_data)
