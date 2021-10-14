import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# year hard coded for now
def compute_league_avg(df, year=2017):
	season = int(str(year) + str(year + 1))
	df_copy = df[df["season"] == season].copy()
	df_copy["coord_tuple"] = df_copy[["x_coordinate_adj", "y_coordinate_adj"]].apply(tuple, axis=1)

	data_league = np.zeros((100, 85))

	for i, j in df_copy['coord_tuple']:
		if np.isnan(i) or np.isnan(j):
			pass
		else:
			data_league[int(i), int(j)] += 1

	# total hours in the season
	season_matchs_drop = df_copy.drop_duplicates(subset=['game_id'],
	                                             keep='last')  # use date to keep the same match with different date
	season_hours = 0
	for i, txt in enumerate(season_matchs_drop['game_time']):
		time = txt.split(':')
		hour_match = int(time[0]) / 60.0 + int(time[1]) / 3600.0
		season_hours += max(hour_match, 1.0)

	data_league = data_league / (
			season_hours * 2)  # need to count each game twice since two team, need to replace with actual calculation of total game time

	# freq_dist = df_copy[["x_coordinate","y_coordinate"]].value_counts().reset_index(name="freq_league")
	# freq_dist["freq_league"] = freq_dist["freq_league"]/1271
	return data_league


# team and year hard coded for now
def compute_team_avg(df, year=2017, team="Colorado Avalanche"):
	season = int(str(year) + str(year + 1))

	df_copy = df[df["season"] == season].copy()
	df_copy2 = df_copy[df_copy['team'] == team].copy()
	df_copy2["coord_tuple"] = df_copy2[["x_coordinate_adj", "y_coordinate_adj"]].apply(tuple, axis=1)

	data_team = np.zeros((100, 85))

	for i, j in df_copy2['coord_tuple']:
		if np.isnan(i) or np.isnan(j):
			pass
		else:
			data_team[int(i), int(j)] += 1

	# count team hours
	# count match as home & away in the season, drop duplicate for detail match
	team_matchs = df_copy.loc[(df_copy["home_team"] == team) | (df_copy['away_team'] == team)]
	team_matchs_drop = team_matchs.drop_duplicates(subset=['game_id'],
	                                               keep='last')  # use date to keep the same match with different date

	team_hours = 0
	for i, txt in enumerate(team_matchs_drop['game_time']):
		time = txt.split(':')
		hour_match = int(time[0]) / 60.0 + int(time[1]) / 3600.0
		team_hours += max(hour_match, 1.0)

	data_team = data_team / team_hours

	return data_team


def all_season_team_avg(df, start_year=2016, end_year=2020, sigma=4, threshold=0.0001):
	team_year = {}
	team_year_shoot = {}
	# start year = 2016 and end year = 2020
	for year in range(start_year, end_year + 1):

		team_year[str(year)] = []
		season = int(str(year) + str(year + 1))

		df_copy = df[df["season"] == season].copy()

		# all teams have been presenting in the season by searching both in home_team and away_team
		all_home_match = df_copy.drop_duplicates(subset=['home_team'], keep='last')  #
		all_away_match = df_copy.drop_duplicates(subset=['away_team'], keep='last')  #
		team_year[str(year)] = np.array(all_home_match['home_team'])
		for _, team in enumerate(np.array(all_away_match['away_team'])):
			if team not in team_year[str(year)]:
				team_year[str(year)].append(team)

	for year in range(start_year, end_year + 1):

		# create a dict for all years and all teams
		# each year includes many teams, each year, team include shoot_frequence array
		team_year_shoot[str(year)] = {}
		test_league = compute_league_avg(tiny, year)
		for team in team_year[str(year)]:
			test_team = compute_team_avg(tiny, year, team)

			test_total = test_team - test_league
			test_total_fine = gaussian_filter(test_total, sigma=sigma)  # smoothing results

			test_total_fine[np.abs(test_total_fine - 0) <= threshold] = None

			team_year_shoot[str(year)][team] = test_total_fine

	return team_year_shoot


def plot_contourf(tiny, year=2016, team="Colorado Avalanche"):
	# team_year_shoot = all_season_team_avg(tiny, start_year = start_year, end_year = end_year)

	test_total = tiny[year][team]
	print('the team name is = {} in the year = {}'.format(team, year))
	test_total = team_year_shoot[str(year)][team]

	xx, yy = np.mgrid[0:100:100j, -42.5:42.5:85j]

	fig = plt.figure()
	ax = fig.gca()
	ax.set_xlim(0, 100)
	ax.set_ylim(-42, 42)
	# plots
	cfset = ax.contourf(xx, yy, test_total, cmap='bwr')
	cset = ax.contour(xx, yy, test_total, colors='k')
	# Label plots
	ax.clabel(cset, inline=1, fontsize=10)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	fig.colorbar(cfset)
	cfset.set_clim(-0.03, 0.03)  # need to find a way to centralise white = 0
	# cset.set_clim(-0.03,0.03)#need to find a way to centralise white = 0
	plt.show()
