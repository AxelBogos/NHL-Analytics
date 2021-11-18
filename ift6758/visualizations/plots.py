import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

TIDY_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tidy_data.csv')
SAVE_FIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')


class HockeyPlotter:
    def __init__(self, fig_size=(15, 10)):
        self.fig_size = fig_size

    def milestone1_shot_type_histogram(self, df, season: str = '20192020', save_fig: bool = True) -> plt.Figure:
        """
        Displays a shot-type histogram as described in Part 5 Question 1
        :param df: tidy pandas.DataFrame
        :param season: string representation of a season, long-formatted (ex: 20192020 for the 2019-2020 season).
        :param save_fig: boolean to save the plot to SAVE_FIG_PATH
        :return: a plt.Figure object instance
        """
        # Filter dataframe by the season of interest
        df = df[df['season'].astype(str) == season]
        fig = plt.figure(figsize=self.fig_size)

        # All shots count plot
        ax1 = sns.countplot(
            x='shot_type',
            data=df,
            order=df['shot_type'].value_counts().index,
            palette=['#7FB5D5'],
            label='shots'
        )

        # Add percentage of shots that are goals per type of shot
        goal_percentage = df[df['is_goal'] == True]['shot_type'].value_counts() / df['shot_type'].value_counts()
        for idx, p in enumerate(ax1.patches):
            height = p.get_height()
            ax1.text(p.get_x() + p.get_width() / 2., height + 450, f'{goal_percentage[idx] * 100 : .2f}%', size=12,
                     ha="center")

        # Goal count plot
        ax2 = sns.countplot(
            x='shot_type',
            data=df[df['is_goal'] == True],
            order=df[df['is_goal'] == True]['shot_type'].value_counts().index,
            palette=['#FFBC7C'],
            label='goals'
        )

        # Format the plot
        plt.legend(labels=['Shots', 'Goals'])
        plt.xticks(rotation=20)
        plt.ylabel('Count of shots')
        plt.xlabel('Type of shot')
        plt.title(
            f'Shot & Goal Count Per Type of Shot and percent of successful goals \n {season[0:4]} - {season[4::]} season')
        ax1.set_axisbelow(True)
        ax1.yaxis.grid(color='gray', linestyle='dashed')
        ax2.set_axisbelow(True)
        ax2.yaxis.grid(color='gray', linestyle='dashed')
        plt.show()
        if save_fig:
            fig.savefig(os.path.join(SAVE_FIG_PATH, 'Q5-1_shot_type_histogram.png'))
        return fig

    def milestone1_distance_vs_goal_chance(self, df, save_fig=True) -> plt.Figure:
        """
        Plots a comparative graph across seasons (2017 - 2020) of the relationship between
        shot distance and goals (as described in Part 5 Q2)
        :param df: tidy pandas.DataFrame
        :param save_fig: boolean to save the plot to SAVE_FIG_PATH
        :return: a plt.Figure object instance
        """

        filtered_df = df[df['shot_distance'].notnull()]
        filtered_df['shot_distance'] = filtered_df['shot_distance'].round(0)
        filtered_df = filtered_df.groupby(["shot_distance", "season"])["is_goal"].mean().to_frame().reset_index()
        fig = plt.figure(figsize=(15, 20))
        # iterate over years to create subplots
        for year_index, year in enumerate(['20182019', '20192020', '20202021']):
            plt.subplot(3, 1, year_index + 1)

            ax = sns.lineplot(x='shot_distance', y='is_goal',
                              data=filtered_df[filtered_df['season'].astype(str) == year])
            plt.title(f'{year[0:4]} - {year[4::]} season')
            ax.set_axisbelow(True)
            ax.yaxis.grid(color='gray', linestyle='dashed')
            plt.xticks(np.arange(0, 210, 10), rotation=20)
            plt.yticks(np.arange(0, 1.2, 0.2))
            plt.xlabel('Shot distance (ft)')
            plt.ylabel('Goal Probability')
        plt.suptitle('Shot Distance vs Goal Probability \n 2018-19, 2019-20, 2020-21 seasons', size=14,
                     y=0.935)
        plt.show()
        if save_fig:
            fig.savefig(os.path.join(SAVE_FIG_PATH, "Q5-2_distance_vs_goal_chance.png"))
        return fig

    def milestone1_distance_and_type_vs_goal(self, df, season: str = '20172018', save_fig=True) -> plt.Figure:
        """
        Depicts the relationship between shot-distance, shot_type and number of goals.
        As described in part 5 question 3
        :param df: tidy pandas.DataFrame
        :param season: string representation of a season, long-formatted (ex: 20192020 for the 2019-2020 season).
        :param save_fig: boolean to save the plot to SAVE_FIG_PATH
        :return: a plt.Figure object instance
        """
        filtered_data = df.dropna(subset=["shot_type", "shot_distance"]).copy()
        filtered_data = filtered_data[filtered_data["season"].astype(str) == season].copy()
        filtered_data.loc["shot_distance"] = filtered_data["shot_distance"].round(0)

        plot_data = filtered_data.groupby(["shot_distance", "shot_type"])["is_goal"].mean().to_frame().reset_index()
        plot_data = plot_data[plot_data["shot_distance"] <= 100]
        plot_data["shot_distance"] = plot_data["shot_distance"].round(0)

        hist_data = filtered_data[["shot_distance", "shot_type", "game_id"]].groupby(
            ["shot_distance", "shot_type"]).count().reset_index().copy()
        hist_data = hist_data[hist_data["shot_distance"] <= 100].copy()
        dict_hist = {}

        for i, shot_type in enumerate(hist_data["shot_type"].unique()):
            dict_hist[i] = hist_data[hist_data["shot_type"] == shot_type][["shot_distance", "game_id"]].copy()

        g = sns.relplot(data=plot_data, x="shot_distance", y="is_goal", col="shot_type",
                        kind="line", linewidth=1, col_wrap=2, ci=0, facet_kws={'sharey': False, 'sharex': False})

        i = 0
        for ax in g.axes.flat:
            ax2 = ax.twinx()
            sns.histplot(data=dict_hist[i], x="shot_distance", bins=100, ax=ax2, alpha=0.1)
            ax.set_xlabel("shot_distance")
            ax.set_ylabel("goal probability")
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax2.set_yticks([10, 20, 30, 40, 50, 60])
            ax2.set_xticks([0, 20, 40, 60, 80, 100])
            ax2.set_ylabel('number of shot')
            i += 1
        plt.subplots_adjust(hspace=0.2, wspace=0.3)
        g.fig.subplots_adjust(top=0.95)
        g.fig.suptitle('Probability of Scoring Based on Shot Distance for Season 2017')
        plt.show()
        if save_fig:
            g.savefig(os.path.join(SAVE_FIG_PATH, "Q5-3_distance_and_type_vs_goal.png"))
        return g

    def milestone2_goal_rate_vs_distance_and_angle(self, df: pd.DataFrame, save_fig=True) -> plt.Figure:
        """
        Plots a comparative graph across seasons (2017 - 2020) of the relationship between
        shot distance and goals (as described in Part 5 Q2)
        :param df: tidy pandas.DataFrame
        :param save_fig: boolean to save the plot to SAVE_FIG_PATH
        :return: a plt.Figure object instance
        """

        fig = plt.figure(figsize=(25, 20))

        plt.subplot(221)
        filtered_df = df[df['shot_distance'].notnull()]
        filtered_df['shot_distance'] = filtered_df['shot_distance'].round(0)
        filtered_df = filtered_df.groupby(["shot_distance"])["is_goal"].mean().to_frame().reset_index()
        ax = sns.lineplot(x='shot_distance', y='is_goal', data=filtered_df)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(0, 210, 10), rotation=20)
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.xlabel('Shot distance (ft)')
        plt.ylabel('Goal Probability')
        plt.title('Goal Probability vs Distance')

        plt.subplot(222)
        ax = sns.histplot(data=df, x='shot_distance', hue='is_goal', multiple='stack')
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(0, 210, 10), rotation=20)
        plt.xlabel('Shot distance (ft)')
        plt.ylabel('Count')
        plt.title('Histogram of shot distance')

        plt.subplot(223)
        filtered_df = df[df['shot_angle'].notnull()]
        filtered_df['shot_angle'] = filtered_df['shot_angle'].round(0)
        filtered_df = filtered_df.groupby(["shot_angle"])["is_goal"].mean().to_frame().reset_index()
        ax = sns.lineplot(x='shot_angle', y='is_goal', data=filtered_df)
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(0, 210, 10), rotation=20)
        plt.yticks(np.arange(0, 1.2, 0.2))
        plt.xlabel('Shot Angle (deg rel. to center-line)')
        plt.ylabel('Goal Probability')
        plt.title('Goal Probability vs Angle')

        plt.subplot(224)
        ax = sns.histplot(data=df, x='shot_angle', hue='is_goal', multiple='stack')
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(0, 190, 10), rotation=20)
        plt.xlabel('Shot Angle (deg rel. to center-line)')
        plt.ylabel('Goal Probability')
        plt.title('Histogram of shot angle')

        plt.show()
        if save_fig:
            fig.savefig(os.path.join(SAVE_FIG_PATH, 'milestone2', "Q2-2-goal_rate.png"))
        return fig

    def milestone2_goal_dist_by_empty_net(self, df: pd.DataFrame, save_fig=True) -> plt.Figure:

        fig = plt.figure(figsize=(20, 25))
        filtered_df = df[df['is_goal'] == True]
        non_empty_net = filtered_df[filtered_df['is_empty_net'] == False]
        non_empty_net_zoomed = non_empty_net[non_empty_net['shot_distance'] >= 100]
        empty_net = filtered_df[filtered_df['is_empty_net'] == True]

        plt.subplot(311)
        ax = sns.histplot(data=non_empty_net, x='shot_distance')
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(0, 210, 10), rotation=20)
        plt.xlabel('Shot distance (ft)')
        plt.ylabel('Count')
        plt.title('Histogram of shot distance for goals (non-empty net)')
        left, bottom, width, height = (100, -50, 100, 500)
        rect = mpatches.Rectangle((left, bottom), width, height,
                                  fill=False,
                                  color="red",
                                  linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(130, 500, 'Zone of interest shown below', fontsize=15)

        plt.subplot(312)
        ax = sns.histplot(data=non_empty_net_zoomed, x='shot_distance')
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(100, 210, 10), rotation=20)
        plt.xlabel('Shot distance (ft)')
        plt.ylabel('Count')
        plt.title(
            'Histogram of shot distance for goals (non-empty net) \n Zoom on the 100ft to 200ft shot distance zone')
        plt.text(130, 16.5, f'Total number of instance for all seasons: {non_empty_net_zoomed.shape[0]}', fontsize=14)

        plt.subplot(313)
        ax = sns.histplot(data=empty_net, x='shot_distance')
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        plt.xticks(np.arange(0, 210, 10), rotation=20)
        plt.xlabel('Shot distance (ft)')
        plt.ylabel('Count')
        plt.title('Histogram of shot distance for goals (empty net)')

        plt.show()
        if save_fig:
            fig.savefig(os.path.join(SAVE_FIG_PATH, 'milestone2', "Q2-3-empty_net_goal_dist.png"))
        return fig


def main():
    df = pd.read_csv(TIDY_DATA_PATH)

    hockey_plotter = HockeyPlotter()

    # ---- Milestone 1 Plots ----
    # # Plot Q5.1
    # hockey_plotter.milestone1_shot_type_histogram(df)
    # # Plot Q5.2
    # hockey_plotter.milestone1_distance_vs_goal_chance(df)
    # # Plot Q5.3
    # hockey_plotter.milestone1_distance_and_type_vs_goal(df)

    # ---- Milestone 2 Plots ----

    # Plot Q2.1
    
    # Plot Q2.2
    hockey_plotter.milestone2_goal_rate_vs_distance_and_angle(df)

    # Plot Q2.3
    hockey_plotter.milestone2_goal_dist_by_empty_net(df)



if __name__ == "__main__":
    main()
