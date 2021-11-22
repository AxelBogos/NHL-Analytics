import os.path
import pandas as pd
from typing import List

TIDY_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tidy_data.csv')
# TODO double-check which seasons we should be using. Milestone2 description seems inconsistent with Milestone 1 data.
DEFAULT_TRAIN_SEASONS = ['20162017', '20172018', '20182019', '20192020']
DEFAULT_TEST_SEASONS = ['20202021']


def load_data(features: List[str], train_val_seasons: List[str] = None, test_season: List[str] = None, target: str = 'is_goal') -> tuple:
    """
    Loads the dataset, drops all but the desired features and target var and returns the train_val_test split.
    :param features: List of features to be used as strings. Ex: ['shot_distance']
    :param train_val_seasons: List of seasons to be used for the train & val sets. Default: DEFAULT_TRAIN_SEASONS
    :param test_season: List of seasons to be used for the test set. Default: DEFAULT_TEST_SEASONS
    :param target: Target feature for classification/prediction.
    :return: X_train, y_train, X_val, y_val, X_test, y_test as tuple
    """
    assert features, 'Must provide training features'
    if train_val_seasons is None:
        train_val_seasons = DEFAULT_TRAIN_SEASONS
    if test_season is None:
        test_season = DEFAULT_TEST_SEASONS
    df = pd.read_csv(TIDY_DATA_PATH)

    # Convert to numeric classes
    df[target] = df[target].astype(int)

    # Split train-val-test by seasons
    train = df[df['season'].astype(str).isin(train_val_seasons)]
    test = df[df['season'].astype(str).isin(test_season)]

    X_train, y_train = train.drop(train.columns.difference(features), axis=1), train[target]
    X_test, y_test = test.drop(test.columns.difference(features), axis=1), test[target]

    return X_train, y_train, X_test, y_test
