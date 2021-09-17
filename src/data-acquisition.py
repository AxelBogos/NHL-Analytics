import requests
import os

SEASONS = ['2017', '2018', '2019', '2020']


def get_season_data(year: str, base_save_file_path: str = '../data/raw/') -> None:
	"""
	:param year: 4-digit desired season year. For example, '2017' for the 2017-2018 season.
	:param base_save_file_path: Basic path from which season data dirs are created. By default, '../data/raw/'
	:return: None. Saves file at desired location
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
	for game in regular_season:
		file_path = os.path.join(base_save_file_path, year, f'{game}.json')

		# Check if file exists already
		if os.path.isfile(file_path):
			continue

		# Request API
		response = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{game}/feed/live/")

		# Write to file
		with open(file_path, 'w') as f:
			f.write(response.text)


def acquire_all_data():
	for year in SEASONS:
		get_season_data(year)


if __name__ == "__main__":
	acquire_all_data()
