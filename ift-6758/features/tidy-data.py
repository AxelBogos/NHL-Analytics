import json
import pandas as pd
import glob
import os
from utils import get_shot_distance


class DataFrameBuilder:
	def __init__(self, base_file_path='../data/raw/', output_dir='../data/'):
		self.base_file_path = base_file_path
		self.output_dir = output_dir
		self.features = ['game_id', 'game_time', 'period', 'period_time', 'team', 'shooter', 'goalie', 'is_goal',
		                 'shot_type', 'x_coordinate', 'y_coordinate', 'is_empty_net', 'strength', 'is_playoff',
		                 'is_home_team','shot_distance']

	def read_json_file(self, file_path) -> dict:
		"""
		Decode the content of a single json file and returns the content as a dict.
		:param file_path: Path to json file
		:return: Parsed json file as dict
		"""
		with open(file_path) as f:
			return json.load(f)

	def read_all_json(self) -> list:
		"""
		Decodes all json file in dir self.base_file_path. Returns content as a list of dicts.
		:return: list of dict
		"""
		json_files = glob.glob(os.path.join(self.base_file_path, '*.json'))
		return [self.read_json_file(file) for file in json_files]

	def parse_game_data(self, json_data) -> list:
		"""
		Parses the required data from 1 json file (i.e. 1 game).
		:param json_data: json data to be parsed
		:return: returns a list of list (game shots/goals)
		"""
		game_data = []
		if 'liveData' not in json_data or \
				'plays' not in json_data['liveData'] or \
				'allPlays' not in json_data['liveData']['plays']:
			return [None] * len(self.features)

		for event in json_data['liveData']['plays']['allPlays']:
			if event['result']['event'] != 'Goal' and event['result']['event'] != 'Shot':
				continue

			game_data.append({
				'game_id':json_data['gamePk'],
				'game_time': f"{(int(event['about']['period']) - 1) * 20 + int(event['about']['periodTime'].split(':')[0])}:{event['about']['periodTime'].split(':')[1]}",
				'period': event['about']['period'],
				'period_time': event['about']['periodTime'],
				'team': event['team']['name'],
				'shooter': event['players'][0]['player']['fullName'],
				'goalie': event['players'][-1]['player']['fullName'],
				'is_goal': True if event['result']['event'] == 'Goal' else False,
				'shot_type': event['result']['secondaryType'] if 'secondaryType' in event['result'] else None,
				'x_coordinate': event['coordinates']['x'] if 'x' in event['coordinates'] else None,
				'y_coordinate': event['coordinates']['y'] if 'y' in event['coordinates'] else None,
				'is_empty_net': event['result']['emptyNet'] if 'emptyNet' in event['result'] else None,
				'strength': event['result']['strength']['name'] if 'strength' in event['result'] else None,
				'is_playoff': json_data['gameData']['game']['type'] == "P",
				'is_home_team': event['team']['id'] == json_data['gameData']['teams']['home']['id'],
				'shot_distance': get_shot_distance(event['coordinates']['x'], event['coordinates']['y'],
				                  event['team']['id'] == json_data['gameData']['teams']['home']['id'],
				                  event['about']['period']) if 'x' in event['coordinates'] and 'y' in event[
					'coordinates'] else None
			})
		return game_data

	def make_dataframe(self) -> None:
		"""
		This function builds the complete data frame by reading all jsons and storing them in a list,
		then parsing the data into a list of list and finally saving the equivalent pd.DataFrame as a .csv
		:return: None
		"""
		json_data = self.read_all_json()
		result = []
		for game in json_data:
			data = self.parse_game_data(game)
			if data == [None] * len(self.features):  # empty row
				continue
			result.extend([i for i in self.parse_game_data(game)])  # quicker than just extend

		result = pd.DataFrame(result, columns=self.features)

		result.to_csv(os.path.join(self.output_dir, 'tidy_data.csv'), index=False)


if __name__ == "__main__":
	builder = DataFrameBuilder('../data/raw')
	builder.make_dataframe()
