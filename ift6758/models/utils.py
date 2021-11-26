import os.path
import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn import feature_extraction
import numpy as np

TIDY_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tidy_data.csv')
# TODO double-check which seasons we should be using. Milestone2 description seems inconsistent with Milestone 1 data.
DEFAULT_TRAIN_SEASONS = ['20162017', '20172018', '20182019', '20192020']
DEFAULT_TEST_SEASONS = ['20202021']


def load_data(features: List[str], train_val_seasons: List[str] = None, test_season: List[str] = None,
              train_val_ratio: float = 0.2, target: str = 'is_goal', use_standard_scaler: bool = True,
              return_as_numpy: bool = False, drop_all_na: bool = True, do_split_val: bool = True,
              convert_bool_to_int=True, one_hot_encode_categoricals = False) -> tuple:
    """
    Loads the dataset, drops all but the desired features and target var and returns the train_val_test split.
    :param features: List of features to be used as strings. Ex: ['shot_distance']
    :param train_val_seasons: List of seasons to be used for the train & val sets. Default: DEFAULT_TRAIN_SEASONS
    :param test_season: List of seasons to be used for the test set. Default: DEFAULT_TEST_SEASONS
    :param train_val_ratio: Ratio of the train and val sets. Default: 0.2
    :param target: Target feature for classification/prediction.
    :param use_standard_scaler: Boolean to determine whether or not to scale features with SkLearn StandardScaler()
    :param return_as_numpy: True to returns datasets as an np.array, False to return as pd.DataFrame/Series
    :param drop_all_na: True to drop all rows with a NAN feature. False to do no such processing
    :param do_split_val: True to split train into train-val.
    :param convert_bool_to_int: If true, converts boolean columns to ints.
    :param one_hot_encode_categoricals: If true, one-hot encods categorical columns.
    :return: X_train, y_train, X_val, y_val, X_test, y_test as tuple
    """
    assert features, 'Must provide training features'
    if train_val_seasons is None:
        train_val_seasons = DEFAULT_TRAIN_SEASONS
    if test_season is None:
        test_season = DEFAULT_TEST_SEASONS
    df = pd.read_csv(TIDY_DATA_PATH)

    if not do_split_val:
        train_val_ratio = 0

    # Convert to numeric classes
    df[target] = df[target].astype(int)

    # Split train-val-test by seasons
    train = df[df['season'].astype(str).isin(train_val_seasons)]
    val = train.sample(frac=train_val_ratio, random_state=0)
    train = train.drop(val.index)
    test = df[df['season'].astype(str).isin(test_season)]

    # drop all NAN rows
    if drop_all_na:
        train = train.dropna(subset=features)
        train = train.reset_index()
        val = val.dropna(subset=features)
        val = val.reset_index()
        test = test.dropna(subset=features)
        test = test.reset_index()

    # Split X, y
    X_train, y_train = train.drop(train.columns.difference(features), axis=1), train[target]
    X_val, y_val = val.drop(val.columns.difference(features), axis=1), val[target]
    X_test, y_test = test.drop(test.columns.difference(features), axis=1), test[target]


    # Scale numeric columns
    if use_standard_scaler:
        NUMERIC_COLS = X_train.select_dtypes([np.number]).columns
        scaler = StandardScaler()
        X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
        if do_split_val:
            X_val[NUMERIC_COLS] = scaler.transform(X_val[NUMERIC_COLS])
        X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    # One-hot encode categorical columns
    if one_hot_encode_categoricals:
        CAT_COLS = X_train.select_dtypes(exclude=["number", "bool_"]).columns
        X_train = pd.get_dummies(data=X_train, columns=CAT_COLS)
        X_val = pd.get_dummies(data=X_val, columns=CAT_COLS)
        X_test = pd.get_dummies(data=X_test, columns=CAT_COLS)

    if convert_bool_to_int:
        BOOL_COLS = X_train.select_dtypes([bool]).columns
        X_train[BOOL_COLS] = X_train[BOOL_COLS].astype(int)
        X_val[BOOL_COLS] = X_val[BOOL_COLS].astype(int)
        X_test[BOOL_COLS] = X_test[BOOL_COLS].astype(int)

    if return_as_numpy:
        X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
        X_val, y_val = X_val.to_numpy(), y_val.to_numpy()
        X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

    if do_split_val:
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        return X_train, y_train, X_test, y_test

