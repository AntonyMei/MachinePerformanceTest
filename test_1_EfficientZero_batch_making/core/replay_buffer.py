import os
import time
import SMOS
import numpy as np
from multiprocessing.managers import BaseManager
from SMOS_utils import RWLock

from core.storage_config import StorageConfig


class ReplayBuffer(object):
    """Reference : DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY
    Algo. 1 and Algo. 2 in Page-3 of (https://arxiv.org/pdf/1803.00933.pdf
    """
    def __init__(self, storage_config: StorageConfig, config=None):
        self.storage_config = storage_config
        self.config = config
        self.batch_size = config.batch_size
        self.keep_ratio = 1

        self.model_index = 0
        self.model_update_interval = 10

        self.buffer = []
        self.priorities = []
        self.game_look_up = []

        self._eps_collected = 0
        self.base_idx = 0
        self._alpha = config.priority_prob_alpha
        self.transition_top = int(config.transition_num * 10 ** 6)
        self.clear_time = 0

        # RW lock for safe access
        # This must be before SMOS client
        self.RW_lock = RWLock()

        # underlying storage
        self.smos_client = SMOS.Client(connection=storage_config.smos_connection)

    def save_pools(self, pools, gap_step):
        # save a list of game histories
        self.RW_lock.writer_enter()
        for (game, priorities) in pools:
            # Only append end game
            # if end_tag:
            self.save_game(game, True, gap_step, priorities)
        self.RW_lock.writer_leave()

    def save_game(self, game, end_tag, gap_steps, priorities=None):
        """Save a game history block
        Parameters
        ----------
        game: Any
            a game history block
        end_tag: bool
            True -> the game is finished. (always True)
        gap_steps: int
            if the game is not finished, we only save the transitions that can be computed
        priorities: list
            the priorities corresponding to the transitions in the game history
        """
        if self.get_total_len() >= self.config.total_transitions:
            return

        if end_tag:
            self._eps_collected += 1
            valid_len = len(game)
        else:
            valid_len = len(game) - gap_steps

        if priorities is None:
            max_prio = self.priorities.max() if self.buffer else 1
            self.priorities = np.concatenate((self.priorities, [max_prio for _ in range(valid_len)] + [0. for _ in range(valid_len, len(game))]))
        else:
            assert len(game) == len(priorities), " priorities should be of same length as the game steps"
            priorities = priorities.copy().reshape(-1)
            # priorities[valid_len:len(game)] = 0.
            self.priorities = np.concatenate((self.priorities, priorities))

        self.buffer.append(game)
        self.game_look_up += [(self.base_idx + len(self.buffer) - 1, step_pos) for step_pos in range(len(game))]

    def get_game(self, idx):
        # return a game
        self.RW_lock.reader_enter()
        game_id, game_pos = self.game_look_up[idx]
        game_id -= self.base_idx
        game = self.buffer[game_id]
        self.RW_lock.reader_leave()
        return game

    def prepare_batch_context(self, batch_size, beta):
        """Prepare a batch context that contains:
        game_lst:               a list of game histories
        game_pos_lst:           transition index in game (relative index)
        indices_lst:            transition index in replay buffer
        weights_lst:            the weight concering the priority
        make_time:              the time the batch is made (for correctly updating replay buffer when data is deleted)
        Parameters
        ----------
        batch_size: int
            batch size
        beta: float
            the parameter in PER for calculating the priority
        """
        self.RW_lock.reader_enter()
        assert beta > 0

        total = self.get_total_len()

        probs = self.priorities ** self._alpha

        probs /= probs.sum()
        # sample data
        indices_lst = np.random.choice(total, batch_size, p=probs, replace=False)

        weights_lst = (total * probs[indices_lst]) ** (-beta)
        weights_lst /= weights_lst.max()

        game_lst = []
        game_pos_lst = []

        for idx in indices_lst:
            game_id, game_pos = self.game_look_up[idx]
            game_id -= self.base_idx
            game = self.buffer[game_id]

            game_lst.append(game)
            game_pos_lst.append(game_pos)

        make_time = [time.time() for _ in range(len(indices_lst))]

        context = (game_lst, game_pos_lst, indices_lst, weights_lst, make_time)
        self.RW_lock.reader_leave()
        return context

    def update_priorities(self, batch_indices, batch_priorities, make_time):
        # update the priorities for data still in replay buffer
        self.RW_lock.reader_enter()
        for i in range(len(batch_indices)):
            if make_time[i] > self.clear_time:
                idx, prio = batch_indices[i], batch_priorities[i]
                self.priorities[idx] = prio
        self.RW_lock.reader_leave()

    def remove_to_fit(self):
        # remove some old data if the replay buffer is full.
        # use lock to avoid data race
        self.RW_lock.writer_enter()

        current_size = self.size()
        total_transition = self.get_total_len()
        if total_transition > self.transition_top:
            index = 0
            for i in range(current_size):
                total_transition -= len(self.buffer[i])
                if total_transition <= self.transition_top * self.keep_ratio:
                    index = i
                    break

            if total_transition >= self.config.batch_size:
                self._remove(index + 1)

        self.RW_lock.writer_leave()

    def _remove(self, num_excess_games):
        # calculate total length
        excess_games_steps = sum([len(game) for game in self.buffer[:num_excess_games]])

        # delete data from smos and replay buffer
        for game in self.buffer[:num_excess_games]:
            game.delete_data(smos_client=self.smos_client)
        del self.buffer[:num_excess_games]
        self.priorities = self.priorities[excess_games_steps:]
        del self.game_look_up[:excess_games_steps]
        self.base_idx += num_excess_games

        self.clear_time = time.time()

    def clear_buffer(self):
        del self.buffer[:]

    def size(self):
        # number of games
        return len(self.buffer)

    def episodes_collected(self):
        # number of collected histories
        return self._eps_collected

    def get_batch_size(self):
        return self.batch_size

    def get_priorities(self):
        return self.priorities

    def get_total_len(self):
        # number of transitions
        return len(self.priorities)


class ReplayBufferManager(BaseManager):
    pass


def start_replay_buffer_server(storage_config: StorageConfig, config):
    """
    Start a replay buffer in current process. Call this method remotely.
    """
    # initialize replay buffer
    replay_buffer = ReplayBuffer(storage_config=storage_config, config=config)
    ReplayBufferManager.register('get_replay_buffer_proxy', callable=lambda: replay_buffer)

    # start server
    replay_buffer_connection = storage_config.replay_buffer_connection
    manager = ReplayBufferManager(address=(replay_buffer_connection.ip,
                                           replay_buffer_connection.port),
                                  authkey=bytes(replay_buffer_connection.authkey))
    server = manager.get_server()
    server.serve_forever()


def get_replay_buffer(storage_config: StorageConfig):
    """
    Get connection to a replay buffer server.
    """
    # get replay_buffer
    ReplayBufferManager.register('get_replay_buffer_proxy')
    replay_buffer_connection = storage_config.replay_buffer_connection
    replay_buffer_manager = ReplayBufferManager(address=(replay_buffer_connection.ip,
                                                         replay_buffer_connection.port),
                                                authkey=bytes(replay_buffer_connection.authkey))
    replay_buffer_connected = False
    while not replay_buffer_connected:
        try:
            replay_buffer_manager.connect()
            replay_buffer_connected = True
        except ConnectionRefusedError:
            time.sleep(1)
    replay_buffer = replay_buffer_manager.get_replay_buffer_proxy()
    return replay_buffer
