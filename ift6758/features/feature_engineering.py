import pandas as pd
import numpy as np

"""
This file contains all feature engineering functions (i.e. features that are not directly extracted from
the JSON files.
"""


def add_home_offensive_side_feature(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Function to create the column determining on which side of the rink the home team is currently attacking.
    -1 if the net the home team scores into is in the negative x-coordinates, +1 if they score in the net
    in the positive x-coordinates.
    :param df: A complete tidy data-frame
    :param inplace: Boolean determining whether the feature is added in place
    :return: A dataframe with the aforementioned column
    """
    if not inplace:
        df = df.copy()
    if 'home_offensive_side' in df.columns:
        return df
    coordinates = df[df['team'] == df['home_team']]
    coordinates = coordinates.groupby(['game_id', 'home_team', 'period'])['x_coordinate'].mean().reset_index()
    coordinates['home_offensive_side'] = np.sign(coordinates['x_coordinate'])
    coordinates = coordinates.drop(['x_coordinate'], axis=1)
    return pd.merge(df, coordinates, on=['game_id', 'home_team', 'period'], how='left')


def add_shot_distance_feature(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Computes the distance between a shooter and the adversary goal net.
    :param df: A complete tidy data-frame
    :param inplace: Boolean determining whether the feature is added in place
    :return: A dataframe with the aforementioned column
    """

    def compute_net_distance(x, y, team, home_team, home_offensive_side):
        """
        Helper function. Computes and returns the distance between the xy shooter coordinates
        and the net they are scoring into based on "home_offensive_side".
        :param x: shooter x-coordinate
        :param y: shooter y-coordinate
        :param team: shooter's team
        :param home_team: Game home team
        :param home_offensive_side: side of the rink the home team is offensive toward in that period.
        :return: distance between the shooter and the net he shoots towards.
        """
        goal_coord = np.array([89, 0])
        if x is None or y is None:
            return None
        if team == home_team:
            goal_coord = home_offensive_side * goal_coord
        else:
            goal_coord = -1 * home_offensive_side * goal_coord
        return np.linalg.norm(np.array([x, y]) - goal_coord)

    if not inplace:
        df = df.copy()
    df['shot_distance'] = df.apply(lambda row: compute_net_distance(
        row['x_coordinate'],
        row['y_coordinate'],
        row['team'],
        row['home_team'],
        row['home_offensive_side']), axis=1)
    return df


def add_shot_angle(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Computes the angle relative to the middle frontal line of the goal..
    :param df: A complete tidy data-frame
    :param inplace: Boolean determining whether the feature is added in place
    :return: A dataframe with the aforementioned column
    """

    def single_shot_angle(x: float, y: float, team: str, home_team: str, home_offensive_side: int) -> float:
        """
        Helper function. Computes and returns the angle between the xy shooter coordinates
        and the net they are scoring into based on "home_offensive_side".
        :param x: shooter x-coordinate
        :param y: shooter y-coordinate
        :param team: shooter's team
        :param home_team: Game home team
        :param home_offensive_side: side of the rink the home team is offensive toward in that period.
        :return: angle between the front of the goal and the shot
        """
        goal_coord = np.array([89, 0])
        if x is None or y is None:
            return 0
        if team == home_team:
            goal_coord = home_offensive_side * goal_coord
        else:
            goal_coord = -1 * home_offensive_side * goal_coord

        relative_x = x - goal_coord[0]  # bring x-coordinate relative to the goal
        angle = 0  # Defaults to 0 if x = [-89 or 89]. That's actually common.
        y += 1e-5  # avoid division by zero
        if np.sign(goal_coord[0]) == -1:  # left goal
            if (np.sign(relative_x)) == 1:  # front of the goal
                angle = np.arctan(np.abs(y) / relative_x)
            elif (np.sign(relative_x)) == -1:  # behind the goal
                angle = np.arctan(np.abs(relative_x) / y) + 90
        elif np.sign(goal_coord[0]) == 1:  # right goal
            if (np.sign(relative_x)) == -1:  # front of the goal
                angle = np.arctan(np.abs(y) / np.abs(relative_x))
            elif (np.sign(relative_x)) == 1:  # behind the goal
                angle = np.arctan(relative_x / y) + 90
        return np.rad2deg(angle)

    if not inplace:
        df = df.copy()
    df['shot_angle'] = df.apply(lambda row: single_shot_angle(
        row['x_coordinate'],
        row['y_coordinate'],
        row['team'],
        row['home_team'],
        row['home_offensive_side']), axis=1)

    return df


def add_change_in_shot_angle(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    **THIS FUNCTION IS REALLY UNOPTIMIZED AND REALLY SLOW**.
    Function to create the column determining on which side of the rink the home team is currently attacking.
    -1 if the net the home team scores into is in the negative x-coordinates, +1 if they score in the net
    in the positive x-coordinates.
    :param df: A complete tidy data-frame
    :param inplace: Boolean determining whether the feature is added in place
    :return: A dataframe with the aforementioned column
    """
    if not inplace:
        df = df.copy()
    for index, row in df.iterrows():
        if not (row['is_rebound'] and row['prev_event_type'] == 'Shot'):
            df.loc[index, 'change_in_angle'] = 0
        else:
            prev_angle = df.iloc[index - 1]['shot_angle']
            df.loc[index, 'change_in_angle'] = np.abs(row['shot_angle'] - prev_angle)
    return df
