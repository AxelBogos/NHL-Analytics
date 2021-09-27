import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TIDY_DATA_PATH = os.path.join('..', 'data', 'tidy_data.csv')
SAVE_FIG_PATH = os.path.join('..', '..', 'figures')


class HockeyPlotter:
	def __init__(self, fig_size=(15, 10), data_path=TIDY_DATA_PATH):
		self.fig_size = fig_size
		self.df = pd.read_csv(data_path)

	def shot_type_histogram(self, save_fig=True) -> plt.Figure:
		"""
		Displays a shot-type histogram as described in Part 5 Question 1
		:param save_fig: boolean to save the plot to SAVE_FIG_PATH
		:return: a plt.Figure object instance
		"""
		fig = plt.figure(figsize=self.fig_size)
		ax1 = sns.countplot(
			x='shot_type',
			data=self.df,
			order=self.df['shot_type'].value_counts().index,
			palette=['#7FB5D5'],
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
			palette=['#FFBC7C'],
			label='goals'
		)
		plt.legend(labels=['Shots', 'Goals'])
		plt.xticks(rotation=20)
		plt.ylabel('Count of shots')
		plt.xlabel('Type of shot')
		plt.title('Shot & Goal Count Per Type of Shot and percent of successful goals')
		ax1.set_axisbelow(True)
		ax1.yaxis.grid(color='gray', linestyle='dashed')
		ax2.set_axisbelow(True)
		ax2.yaxis.grid(color='gray', linestyle='dashed')
		plt.show()
		if save_fig:
			fig.savefig(os.path.join(SAVE_FIG_PATH, 'shot_type_histogram.png'))
		return fig

	def distance_vs_goal_chance(self, save_fig=True) -> plt.Figure:
		"""
		Plots a comparative graph across seasons (2017 - 2020) of the relationship between
		shot distance and goals (as described in Part 5 Q2)
		:param save_fig: boolean to save the plot to SAVE_FIG_PATH
		:return: a plt.Figure object instance
		"""

		self.df['season'] = self.df['game_id'].astype(str).str[0:4]
		filtered_df = self.df[self.df['shot_distance'].notnull()]
		filtered_df['is_goal'].astype(bool)
		fig = plt.figure(figsize=(15, 20))
		# iterate over years to create subplots
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
		if save_fig:
			fig.savefig(os.path.join(SAVE_FIG_PATH, "distance_vs_goal_chance.png"))
		return fig

	def distance_and_type_vs_goal(self, save_fig=True) -> plt.Figure:
		"""
		Depicts the relationship between shot-distance, shot_type and number of goals.
		As described in part 5 question 3
		TODO Work in progress.
		:param save_fig: boolean to save the plot to SAVE_FIG_PATH
		:return: a plt.Figure object instance
		"""
		fig = plt.figure(figsize=self.fig_size)
		ax = sns.violinplot(x="shot_type",
		                    y="shot_distance",
		                    data=self.df,
		                    hue='is_goal',
		                    order=self.df[self.df['is_goal'] == True]['shot_type'].value_counts().index)
		plt.xticks(rotation=20)
		ax.yaxis.grid(color='gray', linestyle='dashed')
		plt.title('Types of shot ordered by number of goals')
		if save_fig:
			fig.savefig(os.path.join(SAVE_FIG_PATH, "distance_and_type_vs_goal.png"))
		plt.show()
		return fig

	def distance_and_type_vs_goalv2(self, save_fig=True) -> plt.Figure:
		"""
		Depicts the relationship between shot-distance, shot_type and number of goals.
		As described in part 5 question 3
		TODO Work in progress.
		:param save_fig: boolean to save the plot to SAVE_FIG_PATH
		:return: a plt.Figure object instance
		"""

		fig = plt.figure(figsize=self.fig_size)
		shot_type_dict = {}
		for index, shot in enumerate(self.df['shot_type'].unique()):
			shot_type_dict[shot] = index
		self.df['shot_type'].replace(shot_type_dict, inplace=True)
		sns.jointplot(data=self.df, x='shot_type', y='shot_distance', hue='is_goal')
		if save_fig:
			fig.savefig(os.path.join(SAVE_FIG_PATH, "distance_and_type_vs_goalv2.png"))
		plt.show()
		return fig

def main():
	hockey_plotter = HockeyPlotter()
	hockey_plotter.shot_type_histogram()
	hockey_plotter.distance_vs_goal_chance()
	hockey_plotter.distance_and_type_vs_goal()
	hockey_plotter.distance_and_type_vs_goalv2()

if __name__ == "__main__":
	main()
