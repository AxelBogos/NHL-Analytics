from ift6758.models.utils import load_data
from dotenv import load_dotenv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import lightgbm as lgbm
load_dotenv()
print(os.environ.get('COMET_API'))

FEATURES = ['shot_type', 'is_empty_net', 'strength', 'time_since_prev_event', 'is_rebound', 'distance_to_prev_event',
            'speed_since_prev_event', 'is_penalty_shot', 'shot_distance', 'shot_angle', 'change_in_angle']
X_train, y_train, X_val, y_val, X_test, y_test = load_data(FEATURES,use_standard_scaler=False)

clf = lgbm.LGBMClassifier()
