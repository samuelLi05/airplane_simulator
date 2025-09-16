import copy
import csv
import os.path
import pickle
from collections import deque, namedtuple
from datetime import datetime
from typing import Dict

import networkx.utils.misc

import gym

from airlift.utils.definitions import ROOT_DIR


def check_path(path):
    path_exists = os.path.exists(path)
    if not path_exists:
        os.makedirs(path)



HistoryFrame = namedtuple("HistoryFrame", ["action", "observation", "reward", "done", "info"])

def obs_equal(obs1, obs2):
    # First, make sure the keys match
    if obs1.keys() != obs2.keys():
        return False
    # If they match, compare the values for each key
    else:
        for k in obs1.keys():
            if type(obs1[k]) != type(obs2[k]):
                return False
            elif isinstance(obs1[k], networkx.Graph):
                if not networkx.utils.misc.graphs_equal(obs1[k], obs2[k]):
                    return False
            elif isinstance(obs1[k], Dict):
                if obs_equal(obs1[k], obs2[k]):  # We can recursively call obs_equal with a nested dictionary
                    return False
            elif obs1[k] != obs2[k]:
                return False

    return True




def assert_histories_equal(history1, history2):
    assert len(history1) == len(history2)

    for item1, item2 in zip(history1, history2):
        assert item1.action == item2.action
        assert item1.reward == item2.reward
        assert item1.done == item2.done
        assert obs_equal(item1.observation, item2.observation)
        assert item1.info == item2.info



# Should we use a PettingZoo wrapper?
class HistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.history = deque(maxlen=15000)
        self.file_name_pkl = None
        self.file_name_csv = None

    def reset(self, seed=None) -> Dict:
        observation = self.env.reset(seed)
        data = HistoryFrame(None, copy.deepcopy(observation), None, None, None)
        self.memorize(data)
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        data = HistoryFrame(copy.deepcopy(action), copy.deepcopy(observation), copy.deepcopy(reward), copy.deepcopy(done), copy.deepcopy(info))
        self.memorize(data)
        return observation, reward, done, info

    def memorize(self, data):
        self.history.append(data)

    def write_to_pkl(self):
        now = datetime.now()
        pkl_path = ROOT_DIR + '/airlift/saved_obs/pkl/'
        check_path(pkl_path)
        dt_string = now.strftime("Date_%m-%d-%Y_Time_%H-%M-%S")
        self.file_name_pkl = ROOT_DIR + '/airlift/saved_obs/pkl/Obs_Wrapper_' + dt_string + str(id(self)) + '.pkl'
        pickle.dump(self.history, open(self.file_name_pkl, 'wb'))

    def write_to_csv(self):
        now = datetime.now()
        csv_path = ROOT_DIR + '/airlift/saved_obs/csv/'
        check_path(csv_path)
        dt_string = now.strftime("Date_%m-%d-%Y_Time_%H-%M-%S")
        self.file_name_csv = ROOT_DIR + '/airlift/saved_obs/csv/Obs_Wrapper_' + dt_string + str(id(self)) + '.csv'
        with open(self.file_name_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(self.history)
