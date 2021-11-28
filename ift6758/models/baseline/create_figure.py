import matplotlib
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.lines as mlines
from cometconf import *
from sklearn.calibration import CalibrationDisplay


def plot_Confusion_matrix(y_valid, y_prediction, score, fig_name):
     cm = metrics.confusion_matrix(y_valid, y_prediction)
     plt.figure(figsize=(9, 9))
     sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
     plt.ylabel('Actual label')
     plt.xlabel('Predicted label')
     all_sample_title = 'Accuracy Score: {0}'.format(score)
     plt.title(all_sample_title, size=15)
     plt.savefig('./figures/' + fig_name + '_cf_matrix.png')
     experiment.log_confusion_matrix(matrix=cm, title=fig_name)
     plt.close()

def plot_roc_curve(roc_data, fig_name):
     for i in roc_data:
          plt.plot(i[0], i[1], label="AUC-" + i[3] + "=" + str(round(i[2],4)))
     plt.ylabel('True Positive Rate')
     plt.xlabel('False Positive Rate')
     plt.legend(loc=4)
     plt.savefig('./figures/' + fig_name + '.png')
     experiment.log_figure('AUC_data')
     plt.close()


# plot GOAL RATE
def cumulative_goal_data(y_valid, y_pred):
     '''
     y_valid: testing label
     y_pred: probability of estimate x_valid
     use ecdfplot in seaborn for estimate y_pred probability
     '''
     y_pred_percentile = 100 * stats.rankdata(y_pred, "min") / len(y_pred)
     test_est = np.array([np.round(y_pred_percentile), y_valid]).T
     df_valid_est = pd.DataFrame(test_est, columns=['model_per', 'is_goal'])

     df_fil = df_valid_est[df_valid_est['is_goal'] == 1]
     return df_fil


def fig_cumulative_goal(list_cumul_data):
     '''
     y_valid: testing label
     y_pred: probability of estimate x_valid
     use ecdfplot in seaborn for estimate y_pred probability
     '''
     plt.figure()
     for i in list_cumul_data:
          ax = sns.ecdfplot(data=i, x=100 - i.model_per, label=i.cat)

     yvals = ax.get_yticks()
     # plt.yticks(np.arange(min(yvals), max(yvals)*1.05, 0.1))
     plt.yticks(np.arange(0, 1.05, 0.1))
     plt.xticks(np.arange(0, 100 * 1.01, 10))
     xvals = ax.get_xticks()
     ax.set_xticklabels(100 - xvals.astype(np.int32))
     yvals = ax.get_yticks()
     ax.set_yticklabels(['{:,.0%}'.format(y) for y in yvals])
     ax.set(xlabel='Shot probability model percentile')
     ax.set_title("Cumulative % of Goals")
     plt.legend(loc='lower right')
     plt.savefig('./figures/cumulative_goal.png')
     experiment.log_figure(figure_name='cumulative_goal', figure=plt)
     plt.close()



def goal_rate_data(y_valid, y_pred):
     '''
     create goal rate figure
     y_valid: testing label
     y_pred: probability of estimate x_valid
     count number of goal, goal+shot
     change xlable, ylabel of the figure
     '''
     # plot GOAL RATE
     test_est = np.array([np.round(y_pred * 100), y_valid]).T
     df_test_est = pd.DataFrame(test_est)
     g = df_test_est.groupby(0)
     # count goals.
     feature_mat = np.array(g.sum())
     # count total of shots + goals
     group_count = np.array(g[[0]].count())
     goal_percentate = feature_mat / group_count  # goal / (goal + shot)
     model_percentage = list(g.groups.keys())
     # convert model_percentage to percentile
     model_percentile = 100 * stats.rankdata(model_percentage, "min") / len(model_percentage)
     goal_rate = np.array([goal_percentate[:, 0], model_percentile])
     df_test_est = pd.DataFrame(goal_rate[:, ::-1].T, columns=['goal_per', 'model_per'])
     xval = 100 - df_test_est.model_per
     return xval, df_test_est


def fig_goal_rate(list_goal_rate):
     plt.figure()
     for i in list_goal_rate:
          ax = sns.lineplot(x=i[0], y=i[1].goal_per, label=str(i[2]))
     yvals = ax.get_yticks()
     plt.yticks(np.arange(0, 1.05, 0.1))
     plt.xticks(np.arange(0, 100 * 1.01, 10))
     ax.set(xlabel='Shot probability model percentile', ylabel="Goals/(Shots + Goals)")
     ax.set(ylim=(0.05, 1.05))
     ax.set(xlim=(0, 110))
     yvals = ax.get_yticks()
     ax.set_yticklabels(['{:,.0%}'.format(y) for y in yvals])
     xvals = ax.get_xticks()
     ax.set_xticklabels(100 - xvals.astype(np.int32))
     ax.set_title("Goal Rate")
     plt.savefig('./figures/goal_rate.png')
     experiment.log_figure(figure_name='goal_rate', figure=plt)
     plt.close()

def calibration_plot(list_calibration_data):
    fig, ax = plt.subplots()
    for i in list_calibration_data:
          CalibrationDisplay.from_predictions(i[0], i[1], n_bins=20, ax=ax, name=i[2])

    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    fig.suptitle('Calibration plot')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability in each bin')
    plt.legend()
    plt.savefig('./figures/calibration.png')
    experiment.log_figure(figure_name='calibration', figure=plt)
    plt.close()
