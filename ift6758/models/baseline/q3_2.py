from sklearn.linear_model import LogisticRegression
from create_figure import *
from sklearn.metrics import accuracy_score
import pickle


def Q2_regression_distance(X_train, y_train, X_val, y_val):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    y_prediction = logisticRegr.predict(X_val)
    predictions_prob = logisticRegr.predict_proba(X_val)
    file_name = "Q3-2_regression_distance_proba.pkl"
    pickle.dump(logisticRegr, open(file_name, "wb"))
    experiment.log_dataset_hash(X_train)
    experiment.log_model('3-2:logistic regression distance with probabilities', 'regression_distance_proba.pkl')
    experiment.add_tag('3_2:logistic-regression-distance-with-probabilities')
    return logisticRegr, y_val, predictions_prob, y_prediction

def Q2_regression_random(X_train, y_train, X_val, y_val):
    np.random.seed(123)
    random_proba = np.random.uniform(low=0.0, high=1.0, size=(len(X_val)))
    f = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
    y_prediction_r = f(random_proba)
    score = accuracy_score(y_val, y_prediction_r)
    print('Q2 - Random score ..:', score)
    experiment.log_metric("Q2_random_distance", score)

    plot_Confusion_matrix(y_val, y_prediction_r, score, 'baseModel_random_regression_dist')
    experiment.log_dataset_hash(X_train)
    experiment.add_tag('3_2:logistic-regression-distance-with-probabilities')
    return "Random", y_val, random_proba, y_prediction_r









