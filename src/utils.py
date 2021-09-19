import numpy as np


def get_shot_distance(x, y, is_home_team, period) -> float:
	"""
	Computes the distance between a shooter and the adversary goal net.
	Assumes the following standard:
	Home-teams score on the "right-side" of the rink at period 1 and 3 (x=100,y=0)
	and on the "left-side" of the rink on period 2. Reverse for away team.
	TODO The above rule does not seem true. Need to find the logic for which team is on which side at period 1
	:param x: x-coordinate of the shooter
	:param y: y-coordinate of the shooter
	:param is_home_team: is the shooter on the home team of the game
	:param period: period as an int (1,2,3)
	:return: euclidean distance between shooter and the adversary goal net
	"""
	#	if ((period % 2 == 1) and is_home_team) or (period % 2 == 0 and not is_home_team):
	#		goal_coord = np.array([89, 0])
	#	else:
	#		goal_coord = np.array([-89, 0])

	# While waiting for actual rules for which team is on which side at period x, just take minimal distance
	# between the 2 goals. Should be mostly true.
	return min(np.linalg.norm(np.array([89, 0]) - np.array([x, y])),
	           np.linalg.norm(np.array([-89, 0]) - np.array([x, y])))
