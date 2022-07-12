import SMOS
import SMOS_utils


class StorageConfig:
    def __init__(self):
        """
        Storage config contains everything needed for accessing storages
        used in EfficientZero_smos.
        """
        # device allocation
        # Note that n DDP trainers will always occupy the first n cards!!
        self.gpu_worker_visible_devices = [1]
        self.data_worker_visible_devices = [0, 2, 3]
        self.test_visible_device = 3
        self.num_training_gpu = 4

        # connection for remote storages
        free_port_list = SMOS_utils.get_local_free_port(4, 6000, 7000)
        self.shared_storage_connection = SMOS.ConnectionDescriptor(ip="localhost", port=free_port_list[0],
                                                                   authkey=b"antony")
        self.replay_buffer_connection = SMOS.ConnectionDescriptor(ip="localhost", port=free_port_list[1],
                                                                  authkey=b"antony")
        self.watchdog_server_connection = SMOS.ConnectionDescriptor(ip="localhost", port=free_port_list[2],
                                                                    authkey=b"antony")
        self.smos_connection = SMOS.ConnectionDescriptor(ip="localhost", port=free_port_list[3],
                                                         authkey=b"antony")

        # name and number of SharedMemoryObject for each storage
        # replay buffer
        # name of replay buffer
        self.replay_buffer_name = "replaybuffer"
        # max number of entries in replay buffer
        self.replay_buffer_capacity = 512
        # max mcts child count, used for determine block size
        self.max_child_visits_count = 128
        # block size list (max_len = 400)
        self.replay_buffer_block_size_list = [4096 * 1024, 128 * (1024 ** 2),
                                              4096 * 1024, self.max_child_visits_count * 4096,
                                              4096 * 1024]
        # name of zombie queue (garbage collection)
        self.zombie_queue_name = "zombie"
        # max number of entries in zombie queue
        self.zombie_queue_capacity = 1024
        # block size of each entry in zombie queue
        self.zombie_queue_block_size = 32

        # mcts storage (cpu -> gpu queue)
        # name of mcts storage
        self.mcts_storage_name = "mcts"
        # number of mcts storages
        self.mcts_storage_count = 1
        # capacity of each mcts storage
        self.mcts_storage_capacity = 30
        # block size of each entry in mcts storage
        self.mcts_block_size_list = [4 * (1024 ** 3), 1 * (1024 ** 3), 4 * (1024 ** 3), 1 * (1024 ** 3),
                                     1 * (1024 ** 3), 4 * (1024 ** 3), 1 * (1024 ** 3), 4 * (1024 ** 3)]

        # batch storage (gpu -> training queue)
        # name of batch storage
        self.batch_storage_name = "batch"
        # number of batch storages
        self.batch_storage_count = 1
        # capacity of each batch storage
        self.batch_storage_capacity_trainer = 8
        self.batch_storage_capacity_worker = 16
        # block size for each entry in batch storage
        self.batch_storage_block_size_list = [4 * (1024 ** 3), 128 * (1024 ** 2), 128 * (1024 ** 2)]

        # xnode storages
        self.trainer_address = "10.200.3.112"  # machine 18
        # batch related
        self.per_worker_batch_sender_count = 8
        self.batch_receiver_count = 8
        self.zmq_batch_ip_trainer = "*"
        self.zmq_batch_port_list_trainer = [i for i in range(5000, 5000 + self.batch_receiver_count)]
        self.zmq_batch_ip_worker = self.trainer_address
        self.zmq_batch_port_list_worker = [i for i in range(10002, 10002 + self.batch_receiver_count)]
        # training signal related
        self.zmq_signal_ip_trainer = "*"
        self.zmq_signal_port_trainer = 5000 + self.batch_receiver_count
        self.zmq_signal_ip_worker = self.trainer_address
        self.zmq_signal_port_worker = 10002 + self.batch_receiver_count
        self.signal_queue_name = "signal_queue"
        self.signal_queue_capacity = 16
        self.signal_queue_block_size_list = [128 * (1024 ** 2)]
        # priority related
        self.zmq_priority_ip_trainer = "*"
        self.zmq_priority_port_trainer = 5001 + self.batch_receiver_count
        self.zmq_priority_ip_worker = self.trainer_address
        self.zmq_priority_port_worker = 10003 + self.batch_receiver_count
        self.priority_queue_name = "priority_queue"
        self.priority_queue_capacity = 64
        self.priority_queue_block_size_list = [128 * (1024 ** 2), 128 * (1024 ** 2), 128 * (1024 ** 2), 128]
