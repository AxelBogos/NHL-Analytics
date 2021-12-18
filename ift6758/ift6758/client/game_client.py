import json
import requests
import time
import logging
import os

logger = logging.getLogger(__name__)
LIVE_RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'live_raw')
GAME_ID_FLAG = 'GAME_ID_FLAG'


class Game_Client:
	def __init__(self, game_id: int,
	             api_url: str = f"https://statsapi.web.nhl.com/api/v1/game/{GAME_ID_FLAG}/feed/live/") -> None:
		self.save_path = LIVE_RAW_DATA_PATH
		self.api_url = api_url.replace(GAME_ID_FLAG, str(game_id))
		self.game_id = game_id
		self.last_event_processed = 0
		self.file_path = os.path.join(self.save_path, f'{game_id}.json')

	def ping_api(self) -> list:
		"""
		Pings the API and returns all the new events since previous ping as a list.
		Save current files under data/live_raw
		Returns: New events as a list

		"""
		# Request API
		response = requests.get(self.api_url)

		first_request = os.path.isfile(self.file_path)

		if first_request:
			# Write to file
			with open(self.file_path, 'w') as f:
				f.write(response.text)
				json_data = json.load(f)
		else:
			with open(self.file_path, 'w') as f:
				json_data = json.load(f)

		new_events = []
		for event_id, event in enumerate(json_data['liveData']['plays']['allPlays'][self.last_event_processed:]):
			if event['result']['event'] not in ('Goal', 'Shot'):
				continue
			new_events.append(event)
			self.last_event_processed = event_id

		return new_events

	def timed_ping_api(self, time_to_run: int = 120.0, time_interval = 60.0)-> list:
		"""
		Pings the API every `time_to_interval` seconds.
		Args:
			time_to_run: Total thing to idly ping the API in seconds. Defaults to 120 seconds
			time_interval: Interval between each API ping. Defaults to 60 seconds

		Returns: A list of all new events collected during `time_to_run`
		"""
		new_events = []
		starttime = time.time()
		while (time.time() - starttime) < time_to_run:
			new_events.extend(self.ping_api())
			time.sleep(time_interval - ((time.time() - starttime) % time_interval))
