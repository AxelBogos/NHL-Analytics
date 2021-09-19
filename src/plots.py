import matplotlib.pyplot as plt
import numpy as np
import seaborn
import seaborn as sns
import pandas as pd
from utils import get_shot_distance


class HockeyPlotter:
	def __init__(self, fig_size=(15, 10), data_path='../data/tidy_data.csv'):
		self.fig_size = fig_size
		self.df = pd.read_csv(data_path)

	def shot_type_histogram(self) -> plt.Figure:
		"""
		Displays a shot-type histogram as described in Part 5 Question 1
		:return: a plt.Figure object instance
		"""
		fig = plt.figure(figsize=self.fig_size)
		ax1 = sns.countplot(
			x='shot_type',
			data=self.df,
			order=self.df['shot_type'].value_counts().index,
			color='blue',
			label='shots'
		)

		goal_percentage = self.df[self.df['is_goal'] == True]['shot_type'].value_counts() / self.df[
			'shot_type'].value_counts()
		for idx, p in enumerate(ax1.patches):
			height = p.get_height()
			ax1.text(p.get_x() + p.get_width() / 2., height + 450, f'{goal_percentage[idx] * 100 : .2f}%',
			         size=12, ha="center")

		ax2 = sns.countplot(
			x='shot_type',
			data=self.df[self.df['is_goal'] == True],
			order=self.df[self.df['is_goal'] == True]['shot_type'].value_counts().index,
			color='red',
			label='goals'
		)
		plt.legend(labels=['Shots', 'Goals'])
		plt.xticks(rotation=20)
		plt.ylabel('Count of shots')
		plt.xlabel('Type of shot')
		plt.title('Shot & Goal Count Per Type of Shot and percent of successful goals')
		ax2.set_axisbelow(True)
		ax2.yaxis.grid(color='gray', linestyle='dashed')
		plt.show()
		return fig

	def distance_vs_goal_chance(self) -> plt.Figure:
		"""

		:return:
		"""

		self.df['season'] = self.df['game_id'].astype(str).str[0:4]
		filtered_df = self.df[self.df['shot_distance'].notnull()]
		filtered_df['is_goal'].astype(bool)
		fig = plt.figure(figsize=(15, 20))
		for year_index, year in enumerate(['2018', '2019', '2020']):
			plt.subplot(3, 1, year_index + 1)
			ax = sns.histplot(
				data=filtered_df[filtered_df['season'] == year],
				x='shot_distance',
				hue='is_goal',
				hue_order=[False, True],
				stat='probability',
				kde=True,
				multiple='stack')
			plt.title(year)
			ax.set_axisbelow(True)
			ax.yaxis.grid(color='gray', linestyle='dashed')
			plt.xticks(np.arange(0, 110, 10), rotation=20)
			plt.legend(['Goal', 'Shot'])
			plt.xlabel('Shot distance (ft)')
		plt.suptitle('Shot Distance vs Probability of Scoring 2018-2020 seasons')
		plt.show()
		return fig


if __name__ == "__main__":
	hockey_plotter = HockeyPlotter()
	hockey_plotter.distance_vs_goal_chance()
