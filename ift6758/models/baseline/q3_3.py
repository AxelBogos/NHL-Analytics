from sklearn.linear_model import LogisticRegression
from create_figure import *
import pickle



def Q3_regre_angle(X_train, y_train, X_val, y_val):
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



