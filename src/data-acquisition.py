import requests
import os


class HockeyDataLoader:
	"""
	Class handling all seasonal data loadings.
	"""

	def __init__(self, season_years=None):
		if season_years is None:
			season_years = ['2017', '2018', '2019', '2020']
		self.SEASONS = season_years

	def get_season_data(self, year: str, base_save_file_path: str = '../data/raw/') -> None:
		"""
		:param year: 4-digit desired season year. For example, '2017' for the 2017-2018 season.
		:param base_save_file_path: Basic path from which season data dirs are created. By default, '../data/raw/'
Min		:return: None. Saves file to base_save_file_path/year/game_id.json
		"""

		# Sanity checks
		assert (base_save_file_path.startswith("../data/raw/"))
		assert (len(year) == 4)
		assert (2016 <= int(year) <= 2021)

		# Form Game IDs
		game_numbers = ["%04d" % x for x in range(1, 1272)]  # 0001, 0002, .... 1271
		regular_season = [f'{year}02{game_number}' for game_number in game_numbers]

		# Make season dir if it does not exist
		if not os.path.isdir(os.path.join(base_save_file_path, year)):
			os.mkdir(os.path.join(base_save_file_path, year))

		# Get games data
		for game_id in regular_season:
			self.get_game_data(game_id, year, base_save_file_path, make_asserts=False)

	def get_game_data(self, game_id: str, year: str, base_save_file_path: str, make_asserts: bool = True) -> None:
		"""
		Get a single game data and save it to base_save_file_path/year/game_id.json
		:param game_id: id of the game. ex: 2017020001
		:param year: 4-digit desired season year. For example, '2017' for the 2017-2018 season.
		:param base_save_file_path: Basic path from which season data dirs are created. By default, '../data/raw/'
		:param make_asserts: boolean to determine whether or not make sanity checks. False if function is called from get_season_data
		:return: None
		"""
		if make_asserts:
			assert (base_save_file_path.startswith("../data/raw/"))
			assert (len(year) == 4)
			assert (2016 <= int(year) <= 2021)

		file_path = os.path.join(base_save_file_path, year, f'{game_id}.json')

		# Check if file exists already
		if os.path.isfile(file_path):
			return

		# Request API
		response = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/")

		# Write to file
		with open(file_path, 'w') as f:
			f.write(response.text)

	def acquire_all_data(self):
		"""
		Fetches data for all seasons contained in self.SEASONS
		:return: None
		"""
		for year in self.SEASONS:
			self.get_season_data(year)


if __name__ == "__main__":
	hockey_data_loader = HockeyDataLoader()
	hockey_data_loader.acquire_all_data()
