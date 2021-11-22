from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from create_figure import *
import pickle


def regression_distance(df):
    tidy = df
    tidy_features = pd.DataFrame(tidy.loc[:, ("shot_distance")])
    tidy_target_goal = pd.DataFrame(tidy.loc[:, ("is_goal")])
    x_train, x_test, y_train, y_test = train_test_split(tidy_features, tidy_target_goal, test_size=0.25, random_state=0)
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)
    predictions_prob = logisticRegr.predict_proba(x_test)
    file_name = "regression_distance_proba.pkl"
    pickle.dump(logisticRegr, open(file_name, "wb"))
    experiment.log_dataset_hash(x_train)
    experiment.log_model('3-2:logistic regression distance with probabilities', 'regression_distance_proba.pkl')
    experiment.add_tag('3_2:logistic-regression-distance-with-probabilities')
    return logisticRegr, x_test, y_test, predictions_prob

def regression_random(df):
    random = pd.DataFrame(np.random.uniform(low=0, high=1, size=(len(df))), columns=['random'])
    tidy = pd.concat([df, random], axis=1)
    tidy = tidy.dropna()
    tidy['random_goal'] = tidy["random"].apply(lambda x: 1 if x > 0.5 else 0)

    tidy_target_r = pd.DataFrame(tidy.loc[:, ("random_goal")])
    tidy_features = pd.DataFrame(tidy.loc[:, ("shot_distance")])
    x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(tidy_features, tidy_target_r, test_size=0.25,
                                                                random_state=0)
    logisticRegr_r = LogisticRegression()
    logisticRegr_r.fit(x_train_r, y_train_r)
    predictions_r = logisticRegr_r.predict(x_test_r)
    predictions_r_prob = logisticRegr_r.predict_proba(x_test_r)
    score_r = logisticRegr_r.score(x_test_r, y_test_r)
    plot_Confusion_matrix(y_test_r, predictions_r, score_r, 'baseModel_random_regression_dist')
    file_name = "regression_random_proba.pkl"
    pickle.dump(logisticRegr_r, open(file_name, "wb"))
    experiment.log_dataset_hash(x_train_r)
    experiment.log_model('3-2:logistic regression distance with probabilities', 'regression_random_proba.pkl')
    experiment.add_tag('3_2:logistic-regression-distance-with-probabilities')
    return logisticRegr_r, x_test_r, y_test_r, predictions_r_prob









