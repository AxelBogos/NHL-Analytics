from sklearn.linear_model import LogisticRegression
from create_figure import *
from cometconf import *
import pickle
from sklearn.metrics import f1_score




def Q1_train_regression_evaluate(X_train, y_train, X_val, y_val):
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    experiment.log_dataset_hash(X_train)
    predictions = logisticRegr.predict(X_val)
    f1score = f1_score(y_val, predictions, average='macro')
    file_name = "regression_default_distance.pkl"
    # save
    pickle.dump(logisticRegr, open(file_name, "wb"))
    score = logisticRegr.score(X_val, y_val)
    experiment.log_metric("Q1_train_acc_regression_distance", score)
    experiment.log_metric("Q1_train_F1_score_regression_distance", f1score)
    print('The accuracy of the logistic regression with the default parameters is: ', score)
    print('The F1 score of the logistic regression with the default parameters is: ', f1score)
    plot_Confusion_matrix(y_val, predictions, score, 'baseModel_regression_dist')
    experiment.log_model('3-1:logistic regression with the default parameters', 'regression_default_distance.pkl')

