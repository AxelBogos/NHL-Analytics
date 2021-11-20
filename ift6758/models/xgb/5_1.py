
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
from create_figure import *

def main():
    url = '../../data/tidy_data.csv'
    feature = ['shot_distance']
    target = ['is_goal']
    fig_name = '../../../figures/5_1_distance'
    model = XGBClassifier(use_label_encoder=False, eval_metric = 'error')

    X, y = get_database(url, feature, target)
    xgboost_basic_feature(X, y, model, fig_name)
    
    feature = ['shot_angle']
    fig_name = '../../../figures/5_1_angle'
    X, y = get_database(url, feature, target)
    xgboost_basic_feature(X, y, model, fig_name)
    
    feature = ['shot_distance', 'shot_angle']
    fig_name = '../../../figures/5_1_dist_angle'
    X, y = get_database(url, feature, target)
    xgboost_basic_feature(X, y, model, fig_name)

if __name__ == "__main__":
    main()