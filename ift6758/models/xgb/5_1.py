
# import figure plot
from create_figure import *

# import utils
import sys

utils_path = os.path.abspath(os.path.join('..'))
sys.path.append(utils_path)
from ift6758.models.utils import *
import os
from dotenv import load_dotenv
import numpy as np
load_dotenv()
COMET_API_KEY = os.getenv('COMET_API_KEY')


def train_basic_xgb(X_train, y_train, X_val):
	model = XGBClassifier(use_label_encoder=False, eval_metric='error')
	model.fit(X_train, y_train)
	y_pred = model.predict_proba(X_val)[:, 1]
	return y_pred


def main():
	'''
    create 4 figures and save and /figures with each feature
    '''

	X_train, y_train, X_val, y_val, _, _ = load_data(
		features=['shot_distance', 'shot_angle'],
		train_val_seasons=DEFAULT_TRAIN_SEASONS,
		test_season=DEFAULT_TEST_SEASONS,
		train_val_ratio=0.2,
		target='is_goal',
		use_standard_scaler=True,
		return_as_dataframes=True,
		drop_all_na=False
	)
	y_pred_vec = []
	FEATURES_LIST = [['shot_distance'], ['shot_angle'], ['shot_distance', 'shot_angle']]

	# All combinations of features
	for features in FEATURES_LIST:
		sub_X_train = X_train[features]
		sub_X_val = X_val[features]
		y_pred = train_basic_xgb(sub_X_train, y_train, sub_X_val)
		y_pred_vec.append(y_pred)

	# Random Baseline
	y_pred_vec.append(np.random.uniform(0, 1, size=y_val.shape[0]))

	#fig_roc_auc(y_val, y_pred_vec)
	fig_cumulative_goal(y_val, y_pred_vec)
	# fig_goal_rate(y_val, y_pred_vec)
	# calibration_fig(y_val, y_pred_vec)


if __name__ == "__main__":
	main()
