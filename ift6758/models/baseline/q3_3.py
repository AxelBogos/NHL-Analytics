from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from create_figure import *
import pickle
from sklearn import preprocessing


def Q3_regre_angle(X_train, y_train, X_val, y_val):
    #tidy = df
    #tidy_features = pd.DataFrame(tidy.loc[:,("shot_angle")])
    #tidy_target_goal = pd.DataFrame(tidy.loc[:, ("is_goal")])
    #x_train, x_valid, y_train, y_valid = train_test_split(tidy_features, tidy_target_goal, test_size=0.2, random_state=0)
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    predictions = logisticRegr.predict(X_val)
    predictions_prob = logisticRegr.predict_proba(X_val)
    score = logisticRegr.score(X_val, y_val)
    plot_Confusion_matrix(y_val, predictions, score, 'baseModel_regression_angel')
    file_name = "regression_angle_proba.pkl"
    pickle.dump(logisticRegr, open(file_name, "wb"))
    experiment.log_dataset_hash(X_train)
    experiment.log_model('3-3:logistic regression distance with probabilities', 'regression_angle_proba.pkl')
    experiment.add_tag('3_3:logistic-regression-distance-with-probabilities')
    return logisticRegr, y_val, predictions_prob, predictions

def Q3_regression_angle_distance(X_train, y_train, X_val, y_val):
    #tidy = df
    #min_max_scaler = preprocessing.MinMaxScaler()
    #tidy[['shot_distance', 'shot_angle']] = min_max_scaler.fit_transform(tidy[['shot_distance', 'shot_angle']])
    #tidy = tidy[['shot_distance', 'shot_angle', 'is_goal']]
    #print('df22 before scaling..:', tidy)

    #tidy_features = pd.DataFrame(tidy.loc[:, ("shot_angle", "shot_distance")])
    #tidy_target_goal = pd.DataFrame(tidy.loc[:, ("is_goal")])
    #x_train, x_test, y_train, y_test = train_test_split(tidy_features, tidy_target_goal, test_size=0.2, random_state=0)
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    predictions = logisticRegr.predict(X_val)
    predictions_prob = logisticRegr.predict_proba(X_val)
    score = logisticRegr.score(X_val, y_val)
    plot_Confusion_matrix(y_val, predictions, score, 'baseModel_regression_angel_distance')
    file_name = "Q3-3_regression_angle_distance_proba.pkl"
    pickle.dump(logisticRegr, open(file_name, "wb"))
    experiment.log_dataset_hash(X_train)
    experiment.log_model('3-3:logistic regression angle and distance with probabilities', 'regression_angle_distance_proba.pkl')
    experiment.add_tag('3_3:logistic-regression-angle-distance-with-probabilities')
    return logisticRegr, y_val, predictions_prob, predictions



