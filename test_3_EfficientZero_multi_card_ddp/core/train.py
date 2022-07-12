import multiprocessing as mp
import os
import pickle
import time
from enum import Enum

import SMOS
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from core.model import get_ddp_model_weights
from core.storage_config import StorageConfig
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast as autocast
from torch.nn import L1Loss
from torch.nn.parallel import DistributedDataParallel as DDP


class PacketType(Enum):
    TARGET_MODEL = 9999
    MODEL = 8888
    LOG = 7777
    SIG_REMOVE_TO_FIT = 6666
    SIG_INCREASE_COUNTER = 5555


class BatchPacket:
    def __init__(self, node_id, batch):
        self.node_id = node_id
        self.batch = batch


class PriorityPacket:
    def __init__(self, indices, new_priority, make_time):
        self.indices = indices
        self.new_priority = new_priority
        self.make_time = make_time


class SignalPacket:
    def __init__(self, packet_type: PacketType, create_steps, data):
        self.packet_type = packet_type
        self.create_steps = create_steps
        self.data = data


def consist_loss_func(f1, f2):
    """Consistency loss function: similarity loss
    Parameters
    """
    f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
    return -(f1 * f2).sum(dim=1)


def adjust_lr(config, optimizer, step_count):
    # adjust learning rate, step lr every lr_decay_steps
    if step_count < config.lr_warm_step:
        lr = config.lr_init * step_count / config.lr_warm_step
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        lr = config.lr_init * config.lr_decay_rate ** ((step_count - config.lr_warm_step) // config.lr_decay_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr


def update_weights(rank, model, batch, optimizer, config, scaler, smos_client, storage_config: StorageConfig,
                   step_count, vis_result=False):
    """update models given a batch data
    Parameters
    ----------
    rank: Any
        DDP trainer rank
    model: Any
        EfficientZero models
    batch: Any
        a batch data inlcudes [inputs_batch, targets_batch]
    scaler: Any
        scaler for torch amp
    vis_result: bool
        True -> log some visualization data in tensorboard (some distributions, values, etc)
    """
    inputs_batch, targets_batch, worker_node_id = batch
    obs_batch_ori, action_batch, mask_batch, indices, weights_lst, make_time = inputs_batch
    target_value_prefix, target_value, target_policy = targets_batch

    # [:, 0: config.stacked_observations * 3,:,:]
    # obs_batch_ori is the original observations in a batch
    # obs_batch is the observation for hat s_t (predicted hidden states from dynamics function)
    # obs_target_batch is the observations for s_t (hidden states from representation function)
    # to save GPU memory usage, obs_batch_ori contains (stack + unroll steps) frames
    obs_batch_ori = torch.from_numpy(obs_batch_ori).to(rank).float() / 255.0
    obs_batch = obs_batch_ori[:, 0: config.stacked_observations * config.image_channel, :, :]
    obs_target_batch = obs_batch_ori[:, config.image_channel:, :, :]

    # do augmentations
    if config.use_augmentation:
        obs_batch = config.transform(obs_batch)
        obs_target_batch = config.transform(obs_target_batch)

    # use GPU tensor
    action_batch = torch.from_numpy(action_batch).to(rank).unsqueeze(-1).long()
    mask_batch = torch.from_numpy(mask_batch).to(rank).float()
    target_value_prefix = torch.from_numpy(target_value_prefix).to(rank).float()
    target_value = torch.from_numpy(target_value).to(rank).float()
    target_policy = torch.from_numpy(target_policy).to(rank).float()
    weights = torch.from_numpy(weights_lst).to(rank).float()

    batch_size = obs_batch.size(0)
    assert batch_size == config.batch_size == target_value_prefix.size(0)
    metric_loss = torch.nn.L1Loss()

    # some logs preparation
    other_log = {}
    other_dist = {}

    other_loss = {
        'l1': -1,
        'l1_1': -1,
        'l1_-1': -1,
        'l1_0': -1,
    }
    for i in range(config.num_unroll_steps):
        key = 'unroll_' + str(i + 1) + '_l1'
        other_loss[key] = -1
        other_loss[key + '_1'] = -1
        other_loss[key + '_-1'] = -1
        other_loss[key + '_0'] = -1

    # transform targets to categorical representation
    transformed_target_value_prefix = config.scalar_transform(target_value_prefix)
    target_value_prefix_phi = config.reward_phi(transformed_target_value_prefix)

    transformed_target_value = config.scalar_transform(target_value)
    target_value_phi = config.value_phi(transformed_target_value)

    if config.amp_type == 'torch_amp':
        with autocast():
            value, _, policy_logits, hidden_state, reward_hidden = model.module.initial_inference(obs_batch, rank)
    else:
        value, _, policy_logits, hidden_state, reward_hidden = model.module.initial_inference(obs_batch, rank)
    scaled_value = config.inverse_value_transform(value)

    if vis_result:
        state_lst = hidden_state.detach().cpu().numpy()

    predicted_value_prefixs = []
    # Note: Following line is just for logging.
    if vis_result:
        predicted_values, predicted_policies = scaled_value.detach().cpu(), torch.softmax(policy_logits,
                                                                                          dim=1).detach().cpu()

    # calculate the new priorities for each transition
    value_priority = L1Loss(reduction='none')(scaled_value.squeeze(-1), target_value[:, 0])
    value_priority = value_priority.data.cpu().numpy() + config.prioritized_replay_eps

    # loss of the first step
    value_loss = config.scalar_value_loss(value, target_value_phi[:, 0])
    policy_loss = -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, 0]).sum(1)
    value_prefix_loss = torch.zeros(batch_size).to(rank)
    consistency_loss = torch.zeros(batch_size).to(rank)

    target_value_prefix_cpu = target_value_prefix.detach().cpu()
    gradient_scale = 1 / config.num_unroll_steps
    # loss of the unrolled steps
    if config.amp_type == 'torch_amp':
        # use torch amp
        with autocast():
            for step_i in range(config.num_unroll_steps):
                # unroll with the dynamics function
                value, value_prefix, policy_logits, hidden_state, reward_hidden = model.module.recurrent_inference(
                    hidden_state, reward_hidden, action_batch[:, step_i])

                beg_index = config.image_channel * step_i
                end_index = config.image_channel * (step_i + config.stacked_observations)

                # consistency loss
                if config.consistency_coeff > 0:
                    # obtain the oracle hidden states from representation function
                    _, _, _, presentation_state, _ = model.module.initial_inference(
                        obs_target_batch[:, beg_index:end_index, :, :], rank)
                    # no grad for the presentation_state branch
                    dynamic_proj = model.module.project(hidden_state, with_grad=True)
                    observation_proj = model.module.project(presentation_state, with_grad=False)
                    temp_loss = consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                    # TODO: remove this log item
                    other_loss['consist_' + str(step_i + 1)] = 1.0
                    consistency_loss += temp_loss

                policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1)
                value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1])
                value_prefix_loss += config.scalar_reward_loss(value_prefix, target_value_prefix_phi[:, step_i])
                # Follow MuZero, set half gradient
                hidden_state.register_hook(lambda grad: grad * 0.5)

                # reset hidden states
                if (step_i + 1) % config.lstm_horizon_len == 0:
                    reward_hidden = (torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(rank),
                                     torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(rank))

                if vis_result:
                    scaled_value_prefixs = config.inverse_reward_transform(value_prefix.detach())
                    scaled_value_prefixs_cpu = scaled_value_prefixs.detach().cpu()

                    predicted_values = torch.cat(
                        (predicted_values, config.inverse_value_transform(value).detach().cpu()))
                    predicted_value_prefixs.append(scaled_value_prefixs_cpu)
                    predicted_policies = torch.cat(
                        (predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                    state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                    key = 'unroll_' + str(step_i + 1) + '_l1'

                    value_prefix_indices_0 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 0)
                    value_prefix_indices_n1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == -1)
                    value_prefix_indices_1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 1)

                    target_value_prefix_base = target_value_prefix_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                    other_loss[key] = metric_loss(scaled_value_prefixs_cpu, target_value_prefix_base)
                    if value_prefix_indices_1.any():
                        other_loss[key + '_1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_1],
                                                             target_value_prefix_base[value_prefix_indices_1])
                    if value_prefix_indices_n1.any():
                        other_loss[key + '_-1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_n1],
                                                              target_value_prefix_base[value_prefix_indices_n1])
                    if value_prefix_indices_0.any():
                        other_loss[key + '_0'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_0],
                                                             target_value_prefix_base[value_prefix_indices_0])
    else:
        for step_i in range(config.num_unroll_steps):
            # unroll with the dynamics function
            value, value_prefix, policy_logits, hidden_state, reward_hidden = model.module.recurrent_inference(
                hidden_state, reward_hidden, action_batch[:, step_i])

            beg_index = config.image_channel * step_i
            end_index = config.image_channel * (step_i + config.stacked_observations)

            # consistency loss
            if config.consistency_coeff > 0:
                # obtain the oracle hidden states from representation function
                _, _, _, presentation_state, _ = model.module.initial_inference(
                    obs_target_batch[:, beg_index:end_index, :, :], rank)
                # no grad for the presentation_state branch
                dynamic_proj = model.module.project(hidden_state, with_grad=True)
                observation_proj = model.module.project(presentation_state, with_grad=False)
                temp_loss = consist_loss_func(dynamic_proj, observation_proj) * mask_batch[:, step_i]

                other_loss['consist_' + str(step_i + 1)] = temp_loss.mean().item()
                consistency_loss += temp_loss

            policy_loss += -(torch.log_softmax(policy_logits, dim=1) * target_policy[:, step_i + 1]).sum(1)
            value_loss += config.scalar_value_loss(value, target_value_phi[:, step_i + 1])
            value_prefix_loss += config.scalar_reward_loss(value_prefix, target_value_prefix_phi[:, step_i])
            # Follow MuZero, set half gradient
            hidden_state.register_hook(lambda grad: grad * 0.5)

            # reset hidden states
            if (step_i + 1) % config.lstm_horizon_len == 0:
                reward_hidden = (torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(rank),
                                 torch.zeros(1, config.batch_size, config.lstm_hidden_size).to(rank))

            if vis_result:
                scaled_value_prefixs = config.inverse_reward_transform(value_prefix.detach())
                scaled_value_prefixs_cpu = scaled_value_prefixs.detach().cpu()

                predicted_values = torch.cat((predicted_values, config.inverse_value_transform(value).detach().cpu()))
                predicted_value_prefixs.append(scaled_value_prefixs_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))
                state_lst = np.concatenate((state_lst, hidden_state.detach().cpu().numpy()))

                key = 'unroll_' + str(step_i + 1) + '_l1'

                value_prefix_indices_0 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 0)
                value_prefix_indices_n1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == -1)
                value_prefix_indices_1 = (target_value_prefix_cpu[:, step_i].unsqueeze(-1) == 1)

                target_value_prefix_base = target_value_prefix_cpu[:, step_i].reshape(-1).unsqueeze(-1)

                other_loss[key] = metric_loss(scaled_value_prefixs_cpu, target_value_prefix_base)
                if value_prefix_indices_1.any():
                    other_loss[key + '_1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_1],
                                                         target_value_prefix_base[value_prefix_indices_1])
                if value_prefix_indices_n1.any():
                    other_loss[key + '_-1'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_n1],
                                                          target_value_prefix_base[value_prefix_indices_n1])
                if value_prefix_indices_0.any():
                    other_loss[key + '_0'] = metric_loss(scaled_value_prefixs_cpu[value_prefix_indices_0],
                                                         target_value_prefix_base[value_prefix_indices_0])
    # ----------------------------------------------------------------------------------
    # weighted loss with masks (some invalid states which are out of trajectory.)
    loss = (config.consistency_coeff * consistency_loss + config.policy_loss_coeff * policy_loss +
            config.value_loss_coeff * value_loss + config.reward_loss_coeff * value_prefix_loss)
    weighted_loss = (weights * loss).mean()

    # backward
    parameters = model.parameters()
    if config.amp_type == 'torch_amp':
        with autocast():
            total_loss = weighted_loss
            total_loss.register_hook(lambda grad: grad * gradient_scale)
    else:
        total_loss = weighted_loss
        total_loss.register_hook(lambda grad: grad * gradient_scale)
    optimizer.zero_grad()

    if config.amp_type == 'none':
        total_loss.backward()
    elif config.amp_type == 'torch_amp':
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
    if config.amp_type == 'torch_amp':
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    # ----------------------------------------------------------------------------------
    # update priority
    new_priority = value_priority
    # replay_buffer.update_priorities(indices, new_priority, make_time)
    smos_client.push_to_object(name=storage_config.priority_queue_name,
                               data=[indices, new_priority, make_time, worker_node_id])

    # packing data for logging
    loss_data = (total_loss.item(), weighted_loss.item(), loss.mean().item(), 0, policy_loss.mean().item(),
                 value_prefix_loss.mean().item(), value_loss.mean().item(), consistency_loss.mean())
    if vis_result:
        reward_w_dist, representation_mean, dynamic_mean, reward_mean = model.module.get_params_mean()
        other_dist['reward_weights_dist'] = reward_w_dist
        other_log['representation_weight'] = representation_mean
        other_log['dynamic_weight'] = dynamic_mean
        other_log['reward_weight'] = reward_mean

        # reward l1 loss
        value_prefix_indices_0 = (target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 0)
        value_prefix_indices_n1 = (target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == -1)
        value_prefix_indices_1 = (target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1) == 1)

        target_value_prefix_base = target_value_prefix_cpu[:, :config.num_unroll_steps].reshape(-1).unsqueeze(-1)

        predicted_value_prefixs = torch.stack(predicted_value_prefixs).transpose(1, 0).squeeze(-1)
        predicted_value_prefixs = predicted_value_prefixs.reshape(-1).unsqueeze(-1)
        other_loss['l1'] = metric_loss(predicted_value_prefixs, target_value_prefix_base)
        if value_prefix_indices_1.any():
            other_loss['l1_1'] = metric_loss(predicted_value_prefixs[value_prefix_indices_1],
                                             target_value_prefix_base[value_prefix_indices_1])
        if value_prefix_indices_n1.any():
            other_loss['l1_-1'] = metric_loss(predicted_value_prefixs[value_prefix_indices_n1],
                                              target_value_prefix_base[value_prefix_indices_n1])
        if value_prefix_indices_0.any():
            other_loss['l1_0'] = metric_loss(predicted_value_prefixs[value_prefix_indices_0],
                                             target_value_prefix_base[value_prefix_indices_0])

        td_data = (new_priority, target_value_prefix.detach().cpu().numpy(), target_value.detach().cpu().numpy(),
                   transformed_target_value_prefix.detach().cpu().numpy(),
                   transformed_target_value.detach().cpu().numpy(),
                   target_value_prefix_phi.detach().cpu().numpy(), target_value_phi.detach().cpu().numpy(),
                   predicted_value_prefixs.detach().cpu().numpy(), predicted_values.detach().cpu().numpy(),
                   target_policy.detach().cpu().numpy(), predicted_policies.detach().cpu().numpy(), state_lst,
                   other_loss, other_log, other_dist)
        priority_data = (weights, indices)
    else:
        td_data, priority_data = None, None

    return loss_data, td_data, priority_data, scaler


def _train(rank, model, target_model, smos_client, config, storage_config: StorageConfig,
           batch_receiver_list=None, signal_publisher_proc=None, priority_publisher_proc=None):
    """training loop
    Parameters
    ----------
    model: Any
        EfficientZero models
    target_model: Any
        EfficientZero models for reanalyzing
    summary_writer: Any
        logging for tensorboard
    """
    # ----------------------------------------------------------------------------------
    model = model.to(rank % storage_config.num_training_gpu)
    model = DDP(model, device_ids=[rank % storage_config.num_training_gpu])
    target_model = target_model.to(rank % storage_config.num_training_gpu)

    optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
                          weight_decay=config.weight_decay)

    scaler = GradScaler()

    model.train()
    target_model.eval()
    # ----------------------------------------------------------------------------------
    # set augmentation tools
    if config.use_augmentation:
        config.set_transforms()

    # Note: the interval of the current model and the target model is between x and 2x. (x = target_model_interval)
    # recent_weights is the param of the target model
    recent_weights = get_ddp_model_weights(ddp_model=model)

    # while loop
    total_time, step_count = 0, 0
    log_interval_counter = 0

    # obtain a batch
    file_name0 = f'./batch/batch{config.batch_size}.part0'
    file_name1 = f'./batch/batch{config.batch_size}.part1'
    file_name2 = f'./batch/batch{config.batch_size}.part2'
    file0 = open(file_name0, 'rb')
    file1 = open(file_name1, 'rb')
    file2 = open(file_name2, 'rb')
    batch0 = pickle.load(file0)
    batch1 = pickle.load(file1)
    batch2 = pickle.load(file2)

    while step_count < config.training_steps + config.last_steps:
        # start time
        start_time = time.time()

        # log
        log_interval_counter += 1

        # remove data if the replay buffer is full. (more data settings)
        if step_count % 1000 == 0 and rank == 0:
            # replay_buffer.remove_to_fit()
            remove_to_fit_packet = SignalPacket(packet_type=PacketType.SIG_REMOVE_TO_FIT,
                                                create_steps=step_count, data=None)
            smos_client.push_to_object(name=storage_config.signal_queue_name, data=[remove_to_fit_packet])

        # recover batch structure (i.e. put obs back into input_batch)
        # input batch, target batch, worker node idx
        batch = [batch0, batch1, batch2]
        batch = [[batch[0]] + batch[1], batch[2], 0]

        # increase step counter in worker
        if rank == 0:
            increase_counter_packet = SignalPacket(packet_type=PacketType.SIG_INCREASE_COUNTER,
                                                   create_steps=step_count, data=None)
            smos_client.push_to_object(name=storage_config.signal_queue_name, data=[increase_counter_packet])
        lr = adjust_lr(config=config, optimizer=optimizer, step_count=step_count)

        # update model for self-play
        if step_count % config.checkpoint_interval == 0 and rank == 0:
            ddp_weights = get_ddp_model_weights(ddp_model=model)
            # shared_storage.set_weights(ddp_weights)
            model_packet = SignalPacket(packet_type=PacketType.MODEL,
                                        create_steps=step_count, data=ddp_weights)
            smos_client.push_to_object(name=storage_config.signal_queue_name, data=[model_packet])

        # update model for reanalyzing
        if step_count % config.target_model_interval == 0 and rank == 0:
            # shared_storage.set_target_weights(recent_weights)
            target_model_packet = SignalPacket(packet_type=PacketType.TARGET_MODEL,
                                               create_steps=step_count, data=recent_weights)
            smos_client.push_to_object(name=storage_config.signal_queue_name, data=[target_model_packet])
            recent_weights = get_ddp_model_weights(ddp_model=model)

        if step_count % config.vis_interval == 0 and rank == 0:
            vis_result = True
        else:
            vis_result = False

        if config.amp_type == 'torch_amp':
            # rank used in update weights should be the id of gpu
            log_data = update_weights(rank=rank % storage_config.num_training_gpu,
                                      model=model, batch=batch, optimizer=optimizer,
                                      config=config, scaler=scaler, vis_result=vis_result,
                                      storage_config=storage_config, smos_client=smos_client,
                                      step_count=step_count)
            scaler = log_data[3]
        else:
            # rank used in update weights should be the id of gpu
            log_data = update_weights(rank=rank % storage_config.num_training_gpu,
                                      model=model, batch=batch, optimizer=optimizer,
                                      config=config, scaler=scaler, vis_result=vis_result,
                                      storage_config=storage_config, smos_client=smos_client,
                                      step_count=step_count)

        if step_count % config.log_interval == 0 and rank == 0:
            # _log(config, step_count, log_data[0:3], model, replay_buffer, lr, shared_storage,
            #      summary_writer, vis_result)
            log_packet = SignalPacket(packet_type=PacketType.LOG, create_steps=step_count,
                                      data=[config, step_count, log_data[0:3], lr, vis_result])
            smos_client.push_to_object(name=storage_config.signal_queue_name, data=[log_packet])

        # increase training step
        # TODO: maybe this barrier can be removed
        dist.barrier()
        step_count += 1

        # save models
        if step_count % config.save_ckpt_interval == 0 and rank == 0:
            model_path = os.path.join(os.getcwd(), 'models', 'model_{}.p'.format(step_count))
            torch.save(model.state_dict(), model_path)

        # log training status
        end_time = time.time()
        if step_count != 1:
            total_time += end_time - start_time
        if step_count % 50 == 0:
            time.sleep(rank * 0.05)
            print('[Trainer {}] batch size={}, loop={}, Avg. Tloop={:.2f}, Lst. Tloop={:.2f}'
                  .format(rank, config.batch_size, step_count, total_time / (step_count - 1), end_time - start_time))
            time.sleep(0.5)
            break

    ddp_weights = get_ddp_model_weights(ddp_model=model)
    # shared_storage.set_weights(ddp_weights)
    if rank == 0:
        model_packet = SignalPacket(packet_type=PacketType.MODEL,
                                    create_steps=step_count, data=ddp_weights)
        smos_client.push_to_object(name=storage_config.signal_queue_name, data=[model_packet])
    return ddp_weights


def start_train(rank, model, target_model, config, storage_config: StorageConfig,
                batch_receiver_list=None, signal_publisher_proc=None, priority_publisher_proc=None):
    """
    Start trainer in current process.
    """
    # get storages
    smos_client = SMOS.Client(connection=storage_config.smos_connection)

    # start trainer
    final_weights = _train(rank=rank, model=model, target_model=target_model, smos_client=smos_client, config=config,
                           storage_config=storage_config, batch_receiver_list=batch_receiver_list,
                           signal_publisher_proc=signal_publisher_proc, priority_publisher_proc=priority_publisher_proc)
    return final_weights


def initialize_trainer(config, model_path=None, local_rank=-1):
    """training process
    Parameters
    ----------
    config: Any
        Atari configuration
    model_path: str
        model path for resuming
        default: train from scratch
    local_rank: Any
        local rank of DDP
    """
    # initialize model
    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    if model_path and local_rank == 0:
        print('resume model from path: ', model_path)
        weights = torch.load(model_path)

        model.load_state_dict(weights)
        target_model.load_state_dict(weights)

    # initialize storage
    storage_config = StorageConfig()
    if local_rank == 0:
        """"""""""""""""""""""""""""""""""""""" Storages """""""""""""""""""""""""""""""""""""""
        ctx = mp.get_context('spawn')

        # start server if it's master trainer
        smos_server = SMOS.Server(connection=storage_config.smos_connection)
        smos_server.start()

        # create storages
        smos_client = SMOS.Client(connection=storage_config.smos_connection)
        # batch storage
        for batch_storage_idx in range(storage_config.batch_storage_count):
            batch_storage_name = storage_config.batch_storage_name + f"{batch_storage_idx}"
            smos_client.create_object(name=batch_storage_name,
                                      max_capacity=storage_config.batch_storage_capacity_trainer,
                                      track_count=4, block_size=storage_config.batch_storage_block_size_list + [128])

        # xnode storages
        smos_client.create_object(name=storage_config.signal_queue_name,
                                  max_capacity=storage_config.signal_queue_capacity,
                                  track_count=1, block_size=storage_config.signal_queue_block_size_list)
        smos_client.create_object(name=storage_config.priority_queue_name,
                                  max_capacity=storage_config.priority_queue_capacity,
                                  track_count=4, block_size=storage_config.priority_queue_block_size_list)

        """"""""""""""""""""""""""""""""""""""" Training """""""""""""""""""""""""""""""""""""""
        # start training
        print(f'************* Begin Testing *************')
        start_train(rank=local_rank, model=model, target_model=target_model, config=config,
                    storage_config=storage_config)
        """"""""""""""""""""""""""""""""""""""" Clean up """""""""""""""""""""""""""""""""""""""
        smos_server.stop()

    else:
        # start training
        start_train(rank=local_rank, model=model, target_model=target_model, config=config,
                    storage_config=storage_config)

    # clean up and return
    dist.destroy_process_group()
    if local_rank == 0:
        print(f'************* Test Finished *************')
