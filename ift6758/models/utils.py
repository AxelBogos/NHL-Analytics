import os.path
import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler

TIDY_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tidy_data.csv')
DEFAULT_TRAIN_SEASONS = ['20152016', '20162017', '20172018', '20182019']
DEFAULT_TEST_SEASONS = ['20192020']


def load_data(features: List[str], train_val_seasons: List[str] = None, test_season: List[str] = None,
              train_val_ratio: float = 0.2, target: str = 'is_goal', use_standard_scaler: bool = True,
              return_as_dataframes: bool = True, drop_all_na: bool = True) -> tuple:
    """
    Loads the dataset, drops all but the desired features and target var and returns the train_val_test split.
    :param features: List of features to be used as strings. Ex: ['shot_distance']
    :param train_val_seasons: List of seasons to be used for the train & val sets. Default: DEFAULT_TRAIN_SEASONS
    :param test_season: List of seasons to be used for the test set. Default: DEFAULT_TEST_SEASONS
    :param train_val_ratio: Ratio of the train and val sets. Default: 0.2
    :param target: Target feature for classification/prediction.
    :param use_standard_scaler: Boolean to determine whether or not to scale features with SkLearn StandardScaler()
    :param return_as_dataframes: True to returns datasets as pd.DataFrame/Series, False to return as np.arrays
    :param drop_all_na: True to drop all rows with a NAN feature. False to do no such processing
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
    train_val = df[df['season'].astype(str).isin(train_val_seasons)]
    val = train_val.sample(frac=train_val_ratio, random_state=123)
    train = train_val.drop(val.index)
    test = df[df['season'].astype(str).isin(test_season)]

    if drop_all_na:
        train = train.dropna(subset=features)
        val = val.dropna(subset=features)
        test = test.dropna(subset=features)

    X_train, y_train = train.drop(train.columns.difference(features), axis=1), train[target]
    X_val, y_val = val.drop(val.columns.difference(features), axis=1), val[target]
    X_test, y_test = test.drop(test.columns.difference(features), axis=1), test[target]

    if use_standard_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    if return_as_dataframes:
        X_train, y_train = pd.DataFrame(data=X_train, columns=features), pd.Series(y_train, name=target)
        X_val, y_val = pd.DataFrame(data=X_val, columns=features), pd.Series(y_val, name=target)
        X_test, y_test = pd.DataFrame(data=X_test, columns=features), pd.Series(y_test, name=target)

    return X_train, y_train, X_val, y_val, X_test, y_test
