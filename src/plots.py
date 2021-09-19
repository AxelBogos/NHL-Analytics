import matplotlib.pyplot as plt
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
			ax1.text(p.get_x() + p.get_width() / 2., height + 450, f'{goal_percentage[idx] * 100 : .2f}%', size=12,
			         ha="center")

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
		# xy coordinates of goal nets
		# period 1 and 3
		p13_home_goal = (100, 0)
		p13_away_goal = (-100, 0)

		# period 2
		p2_home_goal = (-100, 0)
		p2_away_goal = (100, 0)
		filtered_df = self.df[self.df['shot_distance'].notnull()]

		fig = plt.figure(figsize=self.fig_size)
		sns.histplot(data=filtered_df,
		             x=self.df['shot_distance'],
		             hue='is_goal')
		plt.show()


if __name__ == "__main__":
	hockey_plotter = HockeyPlotter()
	hockey_plotter.distance_vs_goal_chance()
