from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from create_figure import *
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import f1_score


def Q2_regression_distance(X_train, y_train, X_val, y_val):
    #tidy = df
    #tidy_features = pd.DataFrame(tidy.loc[:, ("shot_distance")])
    #tidy_target_goal = pd.DataFrame(tidy.loc[:, ("is_goal")])
    #x_train, x_valid, y_train, y_valid = train_test_split(tidy_features, tidy_target_goal, test_size=0.2, random_state=0)
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
    #random = pd.DataFrame(np.random.uniform(low=0, high=1, size=(len(df))), columns=['random'])
    #tidy = pd.concat([df, random], axis=1)
    #tidy = df.dropna()
    #tidy['random_goal'] = tidy["random"].apply(lambda x: 1 if x > 0.5 else 0)

    #tidy_target_r = pd.DataFrame(tidy.loc[:, ("is_goal")])
    #tidy_features = pd.DataFrame(tidy.loc[:, ("shot_distance")])
    #x_train_r, x_valid_r, y_train_r, y_valid_r = train_test_split(tidy_features, tidy_target_r, test_size=0.2,
                                                              #  random_state=0)

    random_proba = np.random.uniform(low=0.0, high=1.0, size=(len(X_val)))
    f = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
    y_prediction_r = f(random_proba)
    score = accuracy_score(y_val, y_prediction_r)
    print('Q2 - Random score ..:', score)
    experiment.log_metric("Q2_random_distance", score)

    plot_Confusion_matrix(y_val, y_prediction_r, score, 'baseModel_random_regression_dist')
    #file_name = "regression_random_proba.pkl"
    #pickle.dump(logisticRegr_r, open(file_name, "wb"))
    experiment.log_dataset_hash(X_train)
    #experiment.log_model('3-2:logistic regression distance with probabilities', 'regression_random_proba.pkl')
    experiment.add_tag('3_2:logistic-regression-distance-with-probabilities')
    return "Random", y_val, random_proba, y_prediction_r









