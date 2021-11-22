from comet_ml import Experiment
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from create_figure import *
from cometconf import *
import pickle




def train_regression_evaluate(df):
    tidy = df
    tidy_features = pd.DataFrame(tidy.loc[:, ("shot_distance")])
    tidy_target = pd.DataFrame(tidy.loc[:, ("is_goal")])
    x_train, x_test, y_train, y_test = train_test_split(tidy_features, tidy_target, test_size=0.25, random_state=0)
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)
    experiment.log_dataset_hash(x_train)
    predictions = logisticRegr.predict(x_test)
    file_name = "regression_default_distance.pkl"
    # save
    pickle.dump(logisticRegr, open(file_name, "wb"))
    score = logisticRegr.score(x_test, y_test)
    experiment.log_metric("Q1_train_acc_regrassion_distance", score)
    print('The accuracy of the logistic regression with the default parameters is: ', score)
    plot_Confusion_matrix(y_test, predictions, score, 'baseModel_regression_dist')
    experiment.log_model('3-1:logistic regression with the default parameters', 'regression_default_distance.pkl')

