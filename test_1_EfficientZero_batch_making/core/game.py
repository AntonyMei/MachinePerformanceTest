import copy
import time

import SMOS
import numpy as np

from core.utils import str_to_arr
from core.storage_config import StorageConfig


class Game:
    def __init__(self, env, action_space_size: int, discount: float, config=None):
        self.env = env
        self.action_space_size = action_space_size
        self.discount = discount
        self.config = config

    def legal_actions(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError()

    def close(self, *args, **kwargs):
        self.env.close(*args, **kwargs)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)


class GameHistory:
    """
    A block of game history from a full trajectories.
    The horizons of Atari games are quite large. Split the whole trajectory into several history blocks.
    """
    def __init__(self, action_space, max_length=200, config=None):
        """
        Parameters
        ----------
        action_space: int
            action space
        max_length: int
            max transition number of the history block
        """
        self.action_space = action_space
        self.max_length = max_length
        self.config = config

        self.stacked_observations = config.stacked_observations
        self.discount = config.discount
        self.action_space_size = config.action_space_size
        self.zero_obs_shape = (config.obs_shape[-2], config.obs_shape[-1], config.image_channel)

        self.child_visits = []
        self.root_values = []

        self.actions = []
        self.obs_history = []
        self.rewards = []

        # for SMOS storage
        self.replay_buffer_name = None
        self.zombie_queue_name = None
        self.entry_idx = None
        self.length = None

    def init(self, init_observations):
        """Initialize a history block, stack the previous stacked_observations frames.
        Parameters
        ----------
        init_observations: list
            list of the stack observations in the previous time steps
        """
        self.child_visits = []
        self.root_values = []

        self.actions = []
        self.obs_history = []
        self.rewards = []
        self.target_values = []
        self.target_rewards = []
        self.target_policies = []

        assert len(init_observations) == self.stacked_observations

        for observation in init_observations:
            self.obs_history.append(copy.deepcopy(observation))

    def pad_over(self, next_block_observations, next_block_rewards, next_block_root_values, next_block_child_visits):
        """To make sure the correction of value targets, we need to add (o_t, r_t, etc) from the next history block
        , which is necessary for the bootstrapped values at the end states of this history block.
        Eg: len = 100; target value v_100 = r_100 + gamma^1 r_101 + ... + gamma^4 r_104 + gamma^5 v_105,
            but r_101, r_102, ... are from the next history block.
        Parameters
        ----------
        next_block_observations: list
            o_t from the next history block
        next_block_rewards: list
            r_t from the next history block
        next_block_root_values: list
            root values of MCTS from the next history block
        next_block_child_visits: list
            root visit count distributions of MCTS from the next history block
        """
        assert len(next_block_observations) <= self.config.num_unroll_steps
        assert len(next_block_child_visits) <= self.config.num_unroll_steps
        assert len(next_block_root_values) <= self.config.num_unroll_steps + self.config.td_steps
        assert len(next_block_rewards) <= self.config.num_unroll_steps + self.config.td_steps - 1

        # notice: next block observation should start from (stacked_observation - 1) in next trajectory
        for observation in next_block_observations:
            self.obs_history.append(copy.deepcopy(observation))

        for reward in next_block_rewards:
            self.rewards.append(reward)

        for value in next_block_root_values:
            self.root_values.append(value)

        for child_visits in next_block_child_visits:
            self.child_visits.append(child_visits)

    def is_full(self):
        # history block is full
        return self.__len__() >= self.max_length

    def legal_actions(self):
        return [_ for _ in range(self.action_space.n)]

    def append(self, action, obs, reward):
        # append a transition tuple
        self.actions.append(action)
        self.obs_history.append(obs)
        self.rewards.append(reward)

    def obs(self, i, extra_len=0, padding=False):
        """To obtain an observation of correct format: o[t, t + stack frames + extra len]
        Parameters
        ----------
        i: int
            time step i
        extra_len: int
            extra len of the obs frames
        padding: bool
            True -> padding frames if (t + stack frames) are out of trajectory
        """
        frames = self.obs_history[i:i + self.stacked_observations + extra_len]
        if padding:
            pad_len = self.stacked_observations + extra_len - len(frames)
            if pad_len > 0:
                pad_frames = [frames[-1] for _ in range(pad_len)]
                frames = np.concatenate((frames, pad_frames))
        if self.config.cvt_string:
            frames = [str_to_arr(obs, self.config.gray_scale) for obs in frames]
        return frames

    def zero_obs(self):
        # return a zero frame
        return [np.zeros(self.zero_obs_shape, dtype=np.uint8) for _ in range(self.stacked_observations)]

    def step_obs(self):
        # return an observation of correct format for model inference
        index = len(self.rewards)
        frames = self.obs_history[index:index + self.stacked_observations]
        if self.config.cvt_string:
            frames = [str_to_arr(obs, self.config.gray_scale) for obs in frames]
        return frames

    def get_targets(self, i):
        # return the value/reward/policy targets at step i
        return self.target_values[i], self.target_rewards[i], self.target_policies[i]

    def game_over(self, smos_client, storage_config: StorageConfig):
        """
        post processing the data when a history block is full
        """
        # convert to numpy
        self.rewards = np.array(self.rewards)
        self.obs_history = np.array(self.obs_history)
        self.actions = np.array(self.actions)
        self.child_visits = np.array(self.child_visits)
        self.root_values = np.array(self.root_values)

        # put into shared memory
        status, entry_idx = smos_client.push_to_object(name=storage_config.replay_buffer_name,
                                                       data=[self.rewards, self.obs_history, self.actions,
                                                             self.child_visits, self.root_values])
        assert status == SMOS.SMOS_SUCCESS
        self.entry_idx = entry_idx
        self.length = len(self.actions)
        self.replay_buffer_name = storage_config.replay_buffer_name
        self.zombie_queue_name = storage_config.zombie_queue_name

        # remove data
        self.rewards = None
        self.obs_history = None
        self.actions = None
        self.child_visits = None
        self.root_values = None

    def restore_data(self, reconstructed_object):
        """
        fetch data from SMOS (zero copy)
        """
        self.rewards, self.obs_history, self.actions, self.child_visits, self.root_values = reconstructed_object

    def delete_data(self, smos_client):
        """
        delete data when removing this history from replaybuffer
        """
        # delete
        status = smos_client.delete_entry(name=self.replay_buffer_name, entry_idx=self.entry_idx)
        if status == SMOS.SMOS_PERMISSION_DENIED:
            smos_client.push_to_object(name=self.zombie_queue_name, data=self.entry_idx)

    def store_search_stats(self, visit_counts, root_value, idx: int = None):
        # store the visit count distributions and value of the root node after MCTS
        sum_visits = sum(visit_counts)
        if idx is None:
            self.child_visits.append([visit_count / sum_visits for visit_count in visit_counts])
            self.root_values.append(root_value)
        else:
            self.child_visits[idx] = [visit_count / sum_visits for visit_count in visit_counts]
            self.root_values[idx] = root_value

    def __len__(self):
        if self.length is not None:
            return self.length
        return len(self.actions)
