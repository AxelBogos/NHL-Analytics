from dotenv import load_dotenv
from xgboost import XGBClassifier
from create_figure import *
# utils_path = os.path.abspath(os.path.join('..'))
# sys.path.append(utils_path)
# from utils import *
from ift6758.models.utils import *
load_dotenv()
COMET_API_KEY = os.getenv('COMET_API_KEY')


def train_basic_xgb(X_train, y_train, X_val):
	model = XGBClassifier()
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
		use_standard_scaler=False,
		drop_all_na=True
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
	model_names = ['Distance', 'Angle', 'Distance + Angle', 'Random']
	fig_number = '5-1'
	fig_roc_auc(y_val, y_pred_vec, fig_number, model_names)
	fig_cumulative_goal(y_val, y_pred_vec, fig_number, model_names)
	fig_goal_rate(y_val, y_pred_vec, fig_number, model_names)
	calibration_fig(y_val, y_pred_vec, fig_number, model_names)


if __name__ == "__main__":
	main()
