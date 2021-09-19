import numpy as np


def get_shot_distance(x, y, is_home_team, period) -> float:
	"""
	Computes the distance between a shooter and the adversary goal net.
	Assumes the following standard:
	Home-teams score on the "right-side" of the rink at period 1 and 3 (x=100,y=0)
	and on the "left-side" of the rink on period 2. Reverse for away team.
	:param x: x-coordinate of the shooter
	:param y: y-coordinate of the shooter
	:param is_home_team: is the shooter on the home team of the game
	:param period: period as an int (1,2,3)
	:return: euclidean distance between shooter and the adversary goal net
	"""
	if ((period == 1 or period == 3) and is_home_team) or (period == 2 and not is_home_team):
		goal_coord = np.array([100, 0])
	else:
		goal_coord = np.array([-100, 0])
	return np.linalg.norm(goal_coord - np.array([x, y]))

