import time
import os
import SMOS
import torch
import numpy as np
import core.ctree.cytree as cytree
import pickle

from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.model import concat_output, concat_output_value
from core.utils import prepare_observation_lst, LinearSchedule, get_gpu_memory

from core.storage_config import StorageConfig
from core.replay_buffer import get_replay_buffer
from core.shared_storage import get_shared_storage
from core.watchdog import get_watchdog_server


class BatchWorker_CPU(object):
    def __init__(self, worker_id, replay_buffer, storage, smos_client, config, storage_config: StorageConfig):
        """CPU Batch Worker for reanalyzing targets, see Appendix.
        Prepare the context concerning CPU overhead
        Parameters
        ----------
        worker_id: int
            id of the worker
        replay_buffer: Any
            Replay buffer
        storage: Any
            The model storage
        smos_client: Any
            Used as queue between cpu worker and gpu worker
        """
        self.worker_id = worker_id
        self.replay_buffer = replay_buffer
        self.storage = storage
        self.smos_client = smos_client
        self.mcts_storage_name = storage_config.mcts_storage_name + f"{worker_id % storage_config.mcts_storage_count}"

        self.config = config
        self.storage_config = storage_config

        self.last_model_index = -1
        self.batch_max_num = 20
        self.beta_schedule = LinearSchedule(config.training_steps + config.last_steps,
                                            initial_p=config.priority_prob_beta, final_p=1.0)

    def _prepare_reward_value_context(self, indices, games, state_index_lst, total_transitions):
        """prepare the context of rewards and values for reanalyzing part
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        total_transitions: int
            number of collected transitions
        """
        zero_obs = games[0].zero_obs()
        config = self.config
        value_obs_lst = []
        # the value is valid or not (out of trajectory)
        value_mask = []
        rewards_lst = []
        traj_lens = []

        td_steps_lst = []
        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)

            # off-policy correction: shorter horizon of td steps
            delta_td = (total_transitions - idx) // config.auto_td_steps
            td_steps = config.td_steps - delta_td
            td_steps = np.clip(td_steps, 1, 5).astype(np.int)

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            game_obs = game.obs(state_index + td_steps, config.num_unroll_steps)
            rewards_lst.append(game.rewards)
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                td_steps_lst.append(td_steps)
                bootstrap_index = current_index + td_steps

                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    beg_index = bootstrap_index - (state_index + td_steps)
                    end_index = beg_index + config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = zero_obs

                value_obs_lst.append(obs)

        reward_value_context = [value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst]
        return reward_value_context

    def _prepare_policy_non_re_context(self, indices, games, state_index_lst):
        """prepare the context of policies for non-reanalyzing part, just return the policy in self-play
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        """
        child_visits = []
        traj_lens = []

        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)

            child_visits.append(game.child_visits)

        policy_non_re_context = [state_index_lst, child_visits, traj_lens]
        return policy_non_re_context

    def _prepare_policy_re_context(self, indices, games, state_index_lst):
        """prepare the context of policies for reanalyzing part
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        """
        zero_obs = games[0].zero_obs()
        config = self.config

        with torch.no_grad():
            # for policy
            policy_obs_lst = []
            policy_mask = []  # 0 -> out of traj, 1 -> new policy
            rewards, child_visits, traj_lens = [], [], []
            for game, state_index in zip(games, state_index_lst):
                traj_len = len(game)
                traj_lens.append(traj_len)
                rewards.append(game.rewards)
                child_visits.append(game.child_visits)
                # prepare the corresponding observations
                game_obs = game.obs(state_index, config.num_unroll_steps)
                for current_index in range(state_index, state_index + config.num_unroll_steps + 1):

                    if current_index < traj_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + config.stacked_observations
                        obs = game_obs[beg_index:end_index]
                    else:
                        policy_mask.append(0)
                        obs = zero_obs
                    policy_obs_lst.append(obs)

        policy_re_context = [policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens]
        return policy_re_context

    def make_batch(self, batch_context, ratio, weights=None):
        """prepare the context of a batch
        reward_value_context:        the context of reanalyzed value targets
        policy_re_context:           the context of reanalyzed policy targets
        policy_non_re_context:       the context of non-reanalyzed policy targets
        inputs_batch:                the inputs of batch
        weights:                     the target model weights
        Parameters
        ----------
        batch_context: Any
            batch context from replay buffer
        ratio: float
            ratio of reanalyzed policy (value is 100% reanalyzed)
        weights: Any
            the target model weights
        """
        # obtain the batch context from replay buffer
        game_lst, game_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context

        # restore data for each game in game_lst
        idx_list = [game.entry_idx for game in game_lst]
        status, handle_batch, reconstructed_batch = self.smos_client.batch_read_from_object(name=self.storage_config.replay_buffer_name,
                                                                                            entry_idx_batch=idx_list)

        for game, reconstructed_object in zip(game_lst, reconstructed_batch):
            game.restore_data(reconstructed_object=reconstructed_object)

        batch_size = len(indices_lst)
        obs_lst, action_lst, mask_lst = [], [], []
        # prepare the inputs of a batch
        for i in range(batch_size):
            game = game_lst[i]
            game_pos = game_pos_lst[i]

            _actions = game.actions[game_pos:game_pos + self.config.num_unroll_steps].tolist()
            # add mask for invalid actions (out of trajectory)
            _mask = [1. for i in range(len(_actions))]
            _mask += [0. for _ in range(self.config.num_unroll_steps - len(_mask))]

            _actions += [np.random.randint(0, game.action_space_size) for _ in range(self.config.num_unroll_steps - len(_actions))]

            # obtain the input observations
            obs_lst.append(game_lst[i].obs(game_pos_lst[i], extra_len=self.config.num_unroll_steps, padding=True))
            action_lst.append(_actions)
            mask_lst.append(_mask)

        re_num = int(batch_size * ratio)
        # formalize the input observations
        obs_lst = prepare_observation_lst(obs_lst)

        # formalize the inputs of a batch
        inputs_batch = [obs_lst, action_lst, mask_lst, indices_lst, weights_lst, make_time_lst]
        for i in range(len(inputs_batch)):
            inputs_batch[i] = np.asarray(inputs_batch[i])

        total_transitions = self.replay_buffer.get_total_len()

        # obtain the context of value targets
        reward_value_context = self._prepare_reward_value_context(indices_lst, game_lst, game_pos_lst, total_transitions)

        # 0:re_num -> reanalyzed policy, re_num:end -> non reanalyzed policy
        # reanalyzed policy
        if re_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self._prepare_policy_re_context(indices_lst[:re_num], game_lst[:re_num], game_pos_lst[:re_num])
        else:
            policy_re_context = None

        # non reanalyzed policy
        if re_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self._prepare_policy_non_re_context(indices_lst[re_num:], game_lst[re_num:], game_pos_lst[re_num:])
        else:
            policy_non_re_context = None

        # re-packaging data for faster transmission (no copy on obs)
        # restore using [[item1] + item2, [item3] + item4, item5, [item6] + item7, item8]
        item1 = np.array(reward_value_context[0])
        item2 = reward_value_context[1:]
        item3 = np.array(policy_re_context[0])
        item4 = policy_re_context[1:]
        item5 = policy_non_re_context
        item6 = inputs_batch[0]
        item7 = inputs_batch[1:]
        item8 = weights

        # push to mcts storage
        while True:
            status, _ = self.smos_client.push_to_object(name=self.mcts_storage_name,
                                                        data=[item1, item2, item3, item4, item5, item6, item7, item8])
            if not status == SMOS.SMOS_SUCCESS:
                time.sleep(0.05)
            else:
                break

        # clean up batch from replay buffer
        self.smos_client.batch_release_entry(object_handle_batch=handle_batch)

    def run(self):
        # start making mcts contexts to feed the GPU batch maker
        start = False
        total_time = 0
        batch_count = 0
        while True:
            # wait for starting
            if not start:
                start = self.storage.get_start_signal()
                time.sleep(1)
                continue

            trained_steps = self.storage.get_counter()

            beta = self.beta_schedule.value(trained_steps)
            # obtain the batch context from replay buffer
            batch_context = self.replay_buffer.prepare_batch_context(self.config.batch_size, beta)
            # break
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                time.sleep(30)
                break

            # This model update delay is necessary for correct update (Otherwise cpu
            # worker may update into old models)
            new_model_index = (trained_steps - 5) // self.config.target_model_interval
            if new_model_index > self.last_model_index:
                self.last_model_index = new_model_index
                target_weights = self.storage.get_target_weights()
            else:
                target_weights = None

            # make batch
            start_time = time.time()
            self.make_batch(batch_context, self.config.revisit_policy_search_rate, weights=target_weights)
            end_time = time.time()
            total_time += end_time - start_time
            batch_count += 1
            if batch_count % 25 == 0:
                print('[CPU Result] Batch size {}, Avg. CPU={:.2f}, Lst. CPU={:.2f}'
                      .format(self.config.batch_size, total_time / batch_count, end_time - start_time))
                break


def start_batch_worker_cpu(worker_id, config, storage_config: StorageConfig):
    """
    Start a CPU batch worker. Call this method remotely.
    """

    # get storages
    replay_buffer = get_replay_buffer(storage_config=storage_config)
    shared_storage = get_shared_storage(storage_config=storage_config)
    smos_client = SMOS.Client(connection=storage_config.smos_connection)

    # start CPU worker
    cpu_worker = BatchWorker_CPU(worker_id=worker_id, replay_buffer=replay_buffer, storage=shared_storage,
                                 smos_client=smos_client, config=config, storage_config=storage_config)
    cpu_worker.run()


class BatchWorker_GPU(object):
    def __init__(self, worker_id, replay_buffer, storage, smos_client, watchdog_server,
                 config, storage_config: StorageConfig):
        """GPU Batch Worker for reanalyzing targets, see Appendix.
        receive the context from CPU maker and deal with GPU overheads
        Parameters
        ----------
        worker_id: int
            id of the worker
        replay_buffer: Any
            Replay buffer
        storage: Any
            The model storage
        batch_storage: Any
            The batch storage (batch queue)
        mcts_storage: Ant
            The mcts-related contexts storage
        """
        self.replay_buffer = replay_buffer
        self.config = config
        self.storage_config = storage_config
        self.worker_id = worker_id

        self.model = config.get_uniform_network()
        self.model.to(config.device)
        self.model.eval()

        self.storage = storage
        self.smos_client = smos_client
        self.watchdog_server = watchdog_server
        self.mcts_storage_name = storage_config.mcts_storage_name + f"{worker_id % storage_config.mcts_storage_count}"
        self.batch_storage_name = storage_config.batch_storage_name + f"{worker_id % storage_config.batch_storage_count}"

        self.last_model_index = 0
        self.empty_loop_count = 0
        self.batch_saved = True

    def _prepare_reward_value(self, reward_value_context):
        """prepare reward and value targets from the context of rewards and values
        """
        value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst = reward_value_context
        device = self.config.device
        batch_size = len(value_obs_lst)

        batch_values, batch_value_prefixs = [], []
        with torch.no_grad():
            value_obs_lst = prepare_observation_lst(value_obs_lst)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self.config.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)
                m_obs = torch.from_numpy(value_obs_lst[beg_index:end_index]).to(device).float() / 255.0
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        m_output = self.model.initial_inference(m_obs)
                else:
                    m_output = self.model.initial_inference(m_obs)
                network_output.append(m_output)

            # concat the output slices after model inference
            if self.config.use_root_value:
                # use the root values from MCTS
                # the root values have limited improvement but require much more GPU actors;
                _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(network_output)
                value_prefix_pool = value_prefix_pool.squeeze().tolist()
                policy_logits_pool = policy_logits_pool.tolist()
                roots = cytree.Roots(batch_size, self.config.action_space_size, self.config.num_simulations)
                noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32).tolist() for _ in range(batch_size)]
                roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
                MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots)

                roots_values = roots.get_values()
                value_lst = np.array(roots_values)
            else:
                # use the predicted values
                value_lst = concat_output_value(network_output)

            # get last state value
            value_lst = value_lst.reshape(-1) * (np.array([self.config.discount for _ in range(batch_size)]) ** td_steps_lst)
            value_lst = value_lst * np.array(value_mask)
            value_lst = value_lst.tolist()

            horizon_id, value_index = 0, 0
            for traj_len_non_re, reward_lst, state_index in zip(traj_lens, rewards_lst, state_index_lst):
                # traj_len = len(game)
                target_values = []
                target_value_prefixs = []

                value_prefix = 0.0
                base_index = state_index
                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_lst[value_index]
                    # for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    for i, reward in enumerate(reward_lst[current_index:bootstrap_index]):
                        value_lst[value_index] += reward * self.config.discount ** i

                    # reset every lstm_horizon_len
                    if horizon_id % self.config.lstm_horizon_len == 0:
                        value_prefix = 0.0
                        base_index = current_index
                    horizon_id += 1

                    if current_index < traj_len_non_re:
                        target_values.append(value_lst[value_index])
                        # Since the horizon is small and the discount is close to 1.
                        # Compute the reward sum to approximate the value prefix for simplification
                        value_prefix += reward_lst[current_index]  # * config.discount ** (current_index - base_index)
                        target_value_prefixs.append(value_prefix)
                    else:
                        target_values.append(0)
                        target_value_prefixs.append(value_prefix)
                    value_index += 1

                batch_value_prefixs.append(target_value_prefixs)
                batch_values.append(target_values)

        batch_value_prefixs = np.asarray(batch_value_prefixs)
        batch_values = np.asarray(batch_values)
        return batch_value_prefixs, batch_values

    def _prepare_policy_re(self, policy_re_context):
        """prepare policy targets from the reanalyzed context of policies
        """
        batch_policies_re = []
        if policy_re_context is None:
            return batch_policies_re

        policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens = policy_re_context
        batch_size = len(policy_obs_lst)
        device = self.config.device

        with torch.no_grad():
            policy_obs_lst = prepare_observation_lst(policy_obs_lst)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self.config.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)

                m_obs = torch.from_numpy(policy_obs_lst[beg_index:end_index]).to(device).float() / 255.0
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        m_output = self.model.initial_inference(m_obs)
                else:
                    m_output = self.model.initial_inference(m_obs)
                network_output.append(m_output)

            _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(network_output)
            value_prefix_pool = value_prefix_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()

            roots = cytree.Roots(batch_size, self.config.action_space_size, self.config.num_simulations)
            noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32).tolist() for _ in range(batch_size)]
            roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
            # do MCTS for a new policy with the recent target model
            MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots)

            roots_distributions = roots.get_distributions()
            policy_index = 0
            for state_index, game_idx in zip(state_index_lst, indices):
                target_policies = []

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]

                    if policy_mask[policy_index] == 0:
                        target_policies.append([0 for _ in range(self.config.action_space_size)])
                    else:
                        # game.store_search_stats(distributions, value, current_index)
                        sum_visits = sum(distributions)
                        policy = [visit_count / sum_visits for visit_count in distributions]
                        target_policies.append(policy)

                    policy_index += 1

                batch_policies_re.append(target_policies)

        batch_policies_re = np.asarray(batch_policies_re)
        return batch_policies_re

    def _prepare_policy_non_re(self, policy_non_re_context):
        """prepare policy targets from the non-reanalyzed context of policies
        """
        batch_policies_non_re = []
        if policy_non_re_context is None:
            return batch_policies_non_re

        state_index_lst, child_visits, traj_lens = policy_non_re_context
        with torch.no_grad():
            # for policy
            policy_mask = []  # 0 -> out of traj, 1 -> old policy
            # for game, state_index in zip(games, state_index_lst):
            for traj_len, child_visit, state_index in zip(traj_lens, child_visits, state_index_lst):
                # traj_len = len(game)
                target_policies = []

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    if current_index < traj_len:
                        target_policies.append(child_visit[current_index])
                        policy_mask.append(1)
                    else:
                        target_policies.append([0 for _ in range(self.config.action_space_size)])
                        policy_mask.append(0)

                batch_policies_non_re.append(target_policies)
        batch_policies_non_re = np.asarray(batch_policies_non_re)
        return batch_policies_non_re

    def _prepare_target_gpu(self, make_batch=False):
        # get batch from cpu
        # input_context = self.mcts_storage.pop()
        status, handle, data = self.smos_client.pop_from_object(name=self.mcts_storage_name)

        # check status
        if not status == SMOS.SMOS_SUCCESS:
            time.sleep(0.05)
            self.empty_loop_count += 1
            return False
        else:
            # restore input context from data
            # restore in the form of [[item1] + item2, [item3] + item4, item5, [item6] + item7, item8]
            input_context = [[data[0]] + data[1], [data[2]] + data[3], data[4], [data[5]] + data[6], data[7]]

            # un-package input context
            reward_value_context, policy_re_context, policy_non_re_context, inputs_batch, target_weights = input_context
            if target_weights is not None:
                self.model.load_state_dict(target_weights)
                self.model.to(self.config.device)
                self.model.eval()

            # target reward, value
            batch_value_prefixes, batch_values = self._prepare_reward_value(reward_value_context)

            # target policy
            batch_policies_re = self._prepare_policy_re(policy_re_context)
            batch_policies_non_re = self._prepare_policy_non_re(policy_non_re_context)
            batch_policies = np.concatenate([batch_policies_re, batch_policies_non_re])

            # package into batch
            # a batch contains the inputs and the targets; inputs is prepared in CPU workers
            # input_batch[0] is obs stored as numpy, single out for zero copy transmission
            targets_batch = [batch_value_prefixes, batch_values, batch_policies]
            batch = [inputs_batch[0], inputs_batch[1:], targets_batch]

            # push into batch storage
            if not self.batch_saved:
                file_name0 = f'./batch/batch{self.config.batch_size}.part0'
                file_name1 = f'./batch/batch{self.config.batch_size}.part1'
                file_name2 = f'./batch/batch{self.config.batch_size}.part2'
                file0 = open(file_name0, 'wb')
                file1 = open(file_name1, 'wb')
                file2 = open(file_name2, 'wb')
                pickle.dump(obj=batch[0], file=file0, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(obj=batch[1], file=file1, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(obj=batch[2], file=file2, protocol=pickle.HIGHEST_PROTOCOL)
                self.batch_saved = True
            self.watchdog_server.increase_reanalyze_batch_count()

            # cleanup
            self.smos_client.free_handle(object_handle=handle)
            return True

    def run(self):
        start = False
        total_time = 0
        batch_count = 0
        while True:
            # waiting for start signal
            if not start:
                start = self.storage.get_start_signal()
                time.sleep(0.1)
                continue

            trained_steps = self.storage.get_counter()
            if trained_steps >= self.config.training_steps + self.config.last_steps:
                time.sleep(30)
                break

            start_time = time.time()
            status = self._prepare_target_gpu()
            end_time = time.time()
            if status:
                total_time += end_time - start_time
                batch_count += 1
                if batch_count % 20 == 0:
                    _, mcts_storage_size = self.smos_client.get_entry_count(name=self.mcts_storage_name)
                    print('[GPU Result] Batch size {}, Avg. GPU={:.2f}, Lst. GPU={:.2f}'
                          .format(self.config.batch_size, total_time / batch_count, end_time - start_time))
                    break


def start_batch_worker_gpu(worker_id, config, storage_config: StorageConfig):
    """
    Start a GPU batch worker. Call this method remotely.
    """
    # set the gpu it resides on
    time.sleep(0.3 * worker_id)
    available_memory_list = get_gpu_memory()
    for i in range(len(available_memory_list)):
        if i not in storage_config.gpu_worker_visible_devices:
            available_memory_list[i] = -1
    max_index = available_memory_list.index(max(available_memory_list))
    used_gpu_idx = storage_config.gpu_worker_visible_devices[worker_id % len(storage_config.gpu_worker_visible_devices)]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(used_gpu_idx)

    # get storages
    replay_buffer = get_replay_buffer(storage_config=storage_config)
    shared_storage = get_shared_storage(storage_config=storage_config)
    smos_client = SMOS.Client(connection=storage_config.smos_connection)
    watchdog_server = get_watchdog_server(storage_config=storage_config)

    # start GPU worker
    gpu_worker = BatchWorker_GPU(worker_id=worker_id, replay_buffer=replay_buffer, storage=shared_storage,
                                 smos_client=smos_client, watchdog_server=watchdog_server, config=config,
                                 storage_config=storage_config)
    gpu_worker.run()
