import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn
import seaborn as sns
import pandas as pd

DATA_PATH = '../data/tidy_data.csv'


def shot_type_histogram() -> plt.Figure:
	"""
	Displays a shot-type histogram as described in Part 5 Question 1
	:return: a plt.Figure object instance
	"""
	df = pd.read_csv(DATA_PATH)
	fig = plt.figure(figsize=(15, 10))
	ax1 = sns.countplot(
		x='shot_type',
		data=df,
		order=df['shot_type'].value_counts().index,
		color='blue',
		label='shots'
	)

	goal_percentage = df[df['is_goal'] == True]['shot_type'].value_counts() / df['shot_type'].value_counts()
	for idx, p in enumerate(ax1.patches):
		height = p.get_height()
		ax1.text(p.get_x() + p.get_width() / 2., height + 450, f'{goal_percentage[idx] * 100 : .2f}%', size=12,
		         ha="center")

	ax2 = sns.countplot(
		x='shot_type',
		data=df[df['is_goal'] == True],
		order=df[df['is_goal'] == True]['shot_type'].value_counts().index,
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
