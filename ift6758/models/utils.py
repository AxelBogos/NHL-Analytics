import os.path
import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn import feature_extraction 

TIDY_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tidy_data.csv')
# TODO double-check which seasons we should be using. Milestone2 description seems inconsistent with Milestone 1 data.
DEFAULT_TRAIN_SEASONS = ['20162017', '20172018', '20182019', '20192020']
DEFAULT_TEST_SEASONS = ['20202021']


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


def feature_preprocessing(feature):
    '''
    feature: list of feature
    processing data, NaN, non-number data
    return dataframe of X, and y
    '''     
    df_X, df_y, _, _, _, _ = load_data(
                                                features = feature,
                                                train_val_seasons = DEFAULT_TRAIN_SEASONS, 
                                                test_season = DEFAULT_TEST_SEASONS,
                                                train_val_ratio = 0, 
                                                target = 'is_goal',
                                                use_standard_scaler = False,
                                                return_as_dataframes= True,
                                                drop_all_na=  False
    )
    
    
    # file NaN (as mention in section 2 NaN = 0)
    df_X = df_X.fillna(0)
    
    # Feature encoding for True, False
    df_X = df_X.replace({True: 1, False: 0})
    
    
    # one-hot encoder with shot_type, prev_event_type
    v = feature_extraction.DictVectorizer(sparse=False)
    # feature encode
    if 'shot_type' in df_X.columns:
        X_shot_type = v.fit_transform(df_X[['shot_type']].to_dict('records'))
        cols_shot = ['s'+str(i) for i in range(X_shot_type.shape[1])]
        df_Xs = pd.DataFrame(X_shot_type, columns = cols_shot, index = df_X.index)
        
        #merge
        df_X = df_X.drop(columns=['shot_type'])
        df_X = pd.concat([df_X, df_Xs], axis = 1)
        
    if 'prev_event_type' in df_X.columns:
        X_prev_type = v.fit_transform(df_X[['prev_event_type']].to_dict('records'))
        cols_prev = ['p'+str(i) for i in range(X_prev_type.shape[1])]
        df_Xp = pd.DataFrame(X_prev_type, columns = cols_prev, index = df_X.index)
        
        #merge 
        df_X = df_X.drop(columns=['prev_event_type'])
        df_X = pd.concat([df_X, df_Xp], axis = 1)

    return df_X, df_y