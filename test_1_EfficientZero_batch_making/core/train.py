import multiprocessing as mp
import time

import SMOS
import torch
from core.reanalyze_worker import start_batch_worker_cpu, start_batch_worker_gpu
from core.replay_buffer import start_replay_buffer_server
from core.selfplay_worker import start_data_worker
from core.shared_storage import start_shared_storage_server
from core.storage_config import StorageConfig
from core.watchdog import start_watchdog
from core.watchdog import start_watchdog_server
from core.xnode import signal_subscriber


def initialize_worker(config, exp_path, model_path=None):
    """
    initialize a worker node
    """
    """"""""""""""""""""""""""""""""""""""" Init """""""""""""""""""""""""""""""""""""""
    # initialize model
    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    if model_path:
        print('resume model from path: ', model_path)
        weights = torch.load(model_path)

        model.load_state_dict(weights)
        target_model.load_state_dict(weights)

    # initialize storage config and multiprocessing context
    storage_config = StorageConfig()
    ctx = mp.get_context('spawn')

    """"""""""""""""""""""""""""""""""""""" Storages """""""""""""""""""""""""""""""""""""""
    # shared storage server
    shared_storage_server = ctx.Process(target=start_shared_storage_server,
                                        args=(storage_config, model, target_model))
    shared_storage_server.start()

    # smos server
    smos_server = SMOS.Server(connection=storage_config.smos_connection)
    smos_server.start()

    # mcts storage and batch storage
    smos_client = SMOS.Client(connection=storage_config.smos_connection)
    for mcts_idx in range(storage_config.mcts_storage_count):
        mcts_storage_name = storage_config.mcts_storage_name + f"{mcts_idx}"
        smos_client.create_object(name=mcts_storage_name, max_capacity=storage_config.mcts_storage_capacity,
                                  track_count=8, block_size=storage_config.mcts_block_size_list)
    for batch_storage_idx in range(storage_config.batch_storage_count):
        batch_storage_name = storage_config.batch_storage_name + f"{batch_storage_idx}"
        smos_client.create_object(name=batch_storage_name, max_capacity=storage_config.batch_storage_capacity_worker,
                                  track_count=3, block_size=storage_config.batch_storage_block_size_list)

    # replay buffer
    smos_client.create_object(name=storage_config.replay_buffer_name,
                              max_capacity=storage_config.replay_buffer_capacity,
                              track_count=5, block_size=storage_config.replay_buffer_block_size_list)
    smos_client.create_object(name=storage_config.zombie_queue_name,
                              max_capacity=storage_config.zombie_queue_capacity,
                              track_count=1, block_size=storage_config.zombie_queue_block_size)
    replay_buffer_server = ctx.Process(target=start_replay_buffer_server,
                                       args=(storage_config, config))
    replay_buffer_server.start()

    # watchdog server
    watchdog_server = ctx.Process(target=start_watchdog_server, args=(storage_config,))
    watchdog_server.start()

    """"""""""""""""""""""""""""""""""""""" Xnode """""""""""""""""""""""""""""""""""""""
    # signal subscriber
    signal_subscriber_proc = ctx.Process(target=signal_subscriber,
                                         args=(config, storage_config))
    signal_subscriber_proc.start()
    time.sleep(0.1)

    """"""""""""""""""""""""""""""""""""""" Workers """""""""""""""""""""""""""""""""""""""
    # data workers
    data_workers = [ctx.Process(target=start_data_worker, args=(rank, config, storage_config))
                    for rank in range(0, config.num_actors)]
    for data_worker in data_workers:
        data_worker.start()
        time.sleep(0.1)

    # cpu workers
    cpu_workers = [ctx.Process(target=start_batch_worker_cpu, args=(worker_idx, config, storage_config))
                   for worker_idx in range(config.cpu_actor)]
    for cpu_worker in cpu_workers:
        cpu_worker.start()
        time.sleep(0.1)

    # gpu workers
    gpu_workers = [ctx.Process(target=start_batch_worker_gpu, args=(worker_idx, config, storage_config))
                   for worker_idx in range(config.gpu_actor)]
    for gpu_worker in gpu_workers:
        gpu_worker.start()
        time.sleep(0.1)

    # watchdog
    watchdog_process = ctx.Process(target=start_watchdog, args=(storage_config, "worker"))
    watchdog_process.start()

    """"""""""""""""""""""""""""""""""""""" Clean up """""""""""""""""""""""""""""""""""""""
    for cpu_worker in cpu_workers:
        cpu_worker.join()
    for gpu_worker in gpu_workers:
        gpu_worker.join()
    for data_worker in data_workers:
        data_worker.terminate()
    watchdog_process.terminate()
    signal_subscriber_proc.terminate()

    # stop servers
    shared_storage_server.terminate()
    replay_buffer_server.terminate()
    watchdog_server.terminate()
    smos_server.stop()
    print(f'************* Test Finished *************')
    print()
