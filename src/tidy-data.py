import json
import pandas as pd
import glob
import os


class DataFrameBuilder:
	def __init__(self, base_file_path='../data/raw/', output_dir='../data/'):
		self.base_file_path = base_file_path
		self.output_dir = output_dir

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
		if 'liveData' not in json_data or 'plays' not in json_data['liveData'] or 'allPlays' not in \
				json_data['liveData']['plays']:
			return [None] * 13
		for event in json_data['liveData']['plays']['allPlays']:
			if event['result']['event'] == 'Goal' or event['result']['event'] == 'Shot':
				game_data.append([
					json_data['gamePk'],
					f"{(int(event['about']['period']) - 1) * 20 + int(event['about']['periodTime'].split(':')[0])}:{event['about']['periodTime'].split(':')[1]}",
					event['about']['period'],
					event['about']['periodTime'],
					event['team']['name'],
					event['players'][0]['player']['fullName'],
					event['players'][-1]['player']['fullName'],
					True if event['result']['event'] == 'Goal' else False,
					event['result']['secondaryType'] if 'secondaryType' in event['result'] else None,
					event['coordinates']['x'] if 'x' in event['coordinates'] else None,
					event['coordinates']['y'] if 'y' in event['coordinates'] else None,
					event['result']['emptyNet'] if 'emptyNet' in event['result'] else None,
					event['result']['strength']['name'] if 'strength' in event['result'] else None,
					json_data['gameData']['game']['type'] == "P"
				])
		return game_data

	def make_dataframe(self) -> None:
		"""

		:return:
		"""
		features = ['game_id', 'game_time', 'period', 'period_time', 'team', 'shooter', 'goalie', 'is_goal',
		            'shot_type', 'x_coordinate', 'y_coordinate', 'is_empty_net', 'strength', 'is_playoff']
		result = pd.DataFrame(
			columns=features)
		json_data = self.read_all_json()
		result = []
		for game in json_data:
			data = self.parse_game_data(game)
			if data == [None] * 13:  # empty row
				continue
			result.extend([i for i in self.parse_game_data(game)])  # quicker than just extend

		result = pd.DataFrame(result, columns=features)

		result.to_csv(os.path.join(self.output_dir, 'tidy_data.csv'), index=False)


if __name__ == "__main__":
	builder = DataFrameBuilder('../data/raw')
	builder.make_dataframe()
