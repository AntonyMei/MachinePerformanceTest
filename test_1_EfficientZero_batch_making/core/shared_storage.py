import os
import time
from multiprocessing.managers import BaseManager
from SMOS_utils import RWLock

from core.storage_config import StorageConfig


class SharedStorage(object):
    def __init__(self, model, target_model):
        """Shared storage for models and others
        Parameters
        ----------
        model: any
            models for self-play (update every checkpoint_interval)
        target_model: any
            models for reanalyzing (update every target_model_interval)
        """
        self.step_counter = 0
        self.test_counter = 0
        self.model = model
        self.target_model = target_model
        self.ori_reward_log = []
        self.reward_log = []
        self.reward_max_log = []
        self.test_dict_log = {}
        self.eps_lengths = []
        self.eps_lengths_max = []
        self.temperature_log = []
        self.visit_entropies_log = []
        self.priority_self_play_log = []
        self.distributions_log = {}
        self.start = False

        # locks for data integrity
        self.model_lock = RWLock()
        self.target_model_lock = RWLock()

    def set_start_signal(self):
        self.start = True

    def get_start_signal(self):
        return self.start

    def get_weights(self):
        self.model_lock.reader_enter()
        model_weights = self.model.get_weights()
        self.model_lock.reader_leave()
        return model_weights

    def set_weights(self, weights):
        self.model_lock.writer_enter()
        result = self.model.set_weights(weights)
        self.model_lock.writer_leave()
        return result

    def get_target_weights(self):
        self.target_model_lock.reader_enter()
        target_model_weights = self.target_model.get_weights()
        self.target_model_lock.reader_leave()
        return target_model_weights

    def set_target_weights(self, weights):
        self.target_model_lock.writer_enter()
        result = self.target_model.set_weights(weights)
        self.target_model_lock.writer_leave()
        return result

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def set_counter(self, val):
        self.step_counter = val

    def set_data_worker_logs(self, eps_len, eps_len_max, eps_ori_reward, eps_reward, eps_reward_max, temperature, visit_entropy, priority_self_play, distributions):
        self.eps_lengths.append(eps_len)
        self.eps_lengths_max.append(eps_len_max)
        self.ori_reward_log.append(eps_ori_reward)
        self.reward_log.append(eps_reward)
        self.reward_max_log.append(eps_reward_max)
        self.temperature_log.append(temperature)
        self.visit_entropies_log.append(visit_entropy)
        self.priority_self_play_log.append(priority_self_play)

        for key, val in distributions.items():
            if key not in self.distributions_log.keys():
                self.distributions_log[key] = []
            self.distributions_log[key] += val

    def add_test_log(self, test_counter, test_dict):
        self.test_counter = test_counter
        for key, val in test_dict.items():
            if key not in self.test_dict_log.keys():
                self.test_dict_log[key] = []
            self.test_dict_log[key].append(val)

    def get_worker_logs(self):
        if len(self.reward_log) > 0:
            ori_reward = sum(self.ori_reward_log) / len(self.ori_reward_log)
            reward = sum(self.reward_log) / len(self.reward_log)
            reward_max = sum(self.reward_max_log) / len(self.reward_max_log)
            eps_lengths = sum(self.eps_lengths) / len(self.eps_lengths)
            eps_lengths_max = sum(self.eps_lengths_max) / len(self.eps_lengths_max)
            temperature = sum(self.temperature_log) / len(self.temperature_log)
            visit_entropy = sum(self.visit_entropies_log) / len(self.visit_entropies_log)
            priority_self_play = sum(self.priority_self_play_log) / len(self.priority_self_play_log)
            distributions = self.distributions_log

            self.ori_reward_log = []
            self.reward_log = []
            self.reward_max_log = []
            self.eps_lengths = []
            self.eps_lengths_max = []
            self.temperature_log = []
            self.visit_entropies_log = []
            self.priority_self_play_log = []
            self.distributions_log = {}

        else:
            ori_reward = None
            reward = None
            reward_max = None
            eps_lengths = None
            eps_lengths_max = None
            temperature = None
            visit_entropy = None
            priority_self_play = None
            distributions = None

        if len(self.test_dict_log) > 0:
            test_dict = self.test_dict_log

            self.test_dict_log = {}
            test_counter = self.test_counter
        else:
            test_dict = None
            test_counter = None

        return ori_reward, reward, reward_max, eps_lengths, eps_lengths_max, test_counter, test_dict, temperature, visit_entropy, priority_self_play, distributions


class SharedStorageManager(BaseManager):
    pass


def start_shared_storage_server(storage_config: StorageConfig, model, target_model):
    """
    Start a shared storage in current process. Call this method remotely.
    """
    # initialize shared storage
    share_storage = SharedStorage(model=model, target_model=target_model)
    SharedStorageManager.register('get_shared_storage_proxy', callable=lambda: share_storage)

    # start server
    shared_storage_connection = storage_config.shared_storage_connection
    manager = SharedStorageManager(address=(shared_storage_connection.ip,
                                            shared_storage_connection.port),
                                   authkey=bytes(shared_storage_connection.authkey))
    server = manager.get_server()
    server.serve_forever()


def get_shared_storage(storage_config: StorageConfig):
    """
    Get connection to a shared storage server.
    """
    SharedStorageManager.register('get_shared_storage_proxy')
    shared_storage_connection = storage_config.shared_storage_connection
    shared_storage_manager = SharedStorageManager(address=(shared_storage_connection.ip,
                                                           shared_storage_connection.port),
                                                  authkey=shared_storage_connection.authkey)
    shared_storage_connected = False
    while not shared_storage_connected:
        try:
            shared_storage_manager.connect()
            shared_storage_connected = True
        except ConnectionRefusedError:
            time.sleep(1)
    shared_storage = shared_storage_manager.get_shared_storage_proxy()
    return shared_storage
