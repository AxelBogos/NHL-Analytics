
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibrationDisplay

FIGURE_PATH = os.path.join(os.path.dirname(__file__), '..','..','..', 'figures')

def fig_roc_auc(y_val, y_pred, fig_name) -> None:
    '''
    y_val: testing label
    y_pred: probability of estimate x_test
    '''
    # plot ROC & AUC
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)
    
    #plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    
    fig_name = fig_name + '_roc_auc.png'
    plt.savefig(os.path.join(FIGURE_PATH, fig_name))
    plt.close()
    return None

# plot GOAL RATE
def fig_cumulative_goal(y_val, y_pred, fig_name) -> None:
    '''
    y_val: testing label
    y_pred: probability of estimate x_test
    use ecdfplot in seaborn for estimate y_pred probability
    '''
    y_pred_percentile = 100*stats.rankdata(y_pred, "min")/len(y_pred)
    test_est = np.array([np.round(y_pred_percentile), y_val]).T
    df_test_est = pd.DataFrame(test_est, columns = ['model_per', 'is_goal'])
    
    df_fil = df_test_est[df_test_est['is_goal'] == 1]
    plt.figure()
    ax = sns.ecdfplot(data=df_fil, x=100-df_fil.model_per)
    yvals = ax.get_yticks()
    #plt.yticks(np.arange(min(yvals), max(yvals)*1.05, 0.1))
    plt.yticks(np.arange(0, 1.05, 0.1))

    plt.xticks(np.arange(0, 100*1.01, 10))
    xvals = ax.get_xticks()
    ax.set_xticklabels(100 - xvals.astype(np.int32))
    
    yvals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(y) for y in yvals])
    ax.set(xlabel='Shot probability model percentile')
    ax.set_title("Cumulative % of Goals")
    
    fig_name = fig_name + '_cumulative_goal.png'
    plt.savefig(os.path.join(FIGURE_PATH, fig_name))


    plt.close()
    return None

def fig_goal_rate(y_val, y_pred, fig_name) -> None:
    '''
    create goal rate figure
    y_val: testing label
    y_pred: probability of estimate x_test
    count number of goal, goal+shot
    change xlable, ylabel of the figure
    '''
    # plot GOAL RATE
    test_est = np.array([np.round(y_pred*100), y_val]).T
    df_test_est = pd.DataFrame(test_est)
    
    g = df_test_est.groupby(0)
    
    # count goals.
    feature_mat = np.array(g.sum())
            
    # count total of shots + goals
    group_count = np.array(g[[0]].count())
    
    
    goal_percentate = feature_mat / group_count # goal / (goal + shot)
    model_percentage = list(g.groups.keys())

    # convert model_percentage to percentile
    model_percentile = 100*stats.rankdata(model_percentage, "min")/len(model_percentage)
    
    goal_rate = np.array([goal_percentate[:,0], model_percentile])
    
    df_test_est = pd.DataFrame(goal_rate[:,::-1].T, columns = ['goal_per', 'model_per'])
    
    xval = 100 - df_test_est.model_per
    plt.figure()
    ax = sns.lineplot(x = xval, y = df_test_est.goal_per)
    
    yvals = ax.get_yticks()
    #plt.yticks(np.arange(min(yvals), max(yvals)*1.05, 0.1))
    plt.yticks(np.arange(0, 1.05, 0.1))
        
        
    plt.xticks(np.arange(0, 100*1.01, 10))
    
    
    ax.set(xlabel='Shot probability model percentile', ylabel="Goals/(Shots + Goals)")
    
    ax.set(ylim=(0.05, 1.05))
    ax.set(xlim=(0, 110))
    
    yvals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(y) for y in yvals])
    xvals = ax.get_xticks()
    
    ax.set_xticklabels(100 - xvals.astype(np.int32))
    ax.set_title("Goal Rate")
    
    fig_name = fig_name + '_goal_rate.png'
    plt.savefig(os.path.join(FIGURE_PATH, fig_name))

    plt.close()
    return None
    
def calibration_fig(y_val, y_pred, fig_name) -> None:
    '''
    create calibration curve figures from y_val, y_pred database
    '''

    plt.figure()
    disp = CalibrationDisplay.from_predictions(y_val, y_pred)

    fig_name = fig_name + '_calibration.png'
    plt.savefig(os.path.join(FIGURE_PATH, fig_name))

    plt.close()
    return None
