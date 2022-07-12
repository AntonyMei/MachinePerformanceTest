import time

from core.replay_buffer import get_replay_buffer
from core.shared_storage import get_shared_storage
from core.storage_config import StorageConfig


def signal_subscriber(config, storage_config: StorageConfig):
    # get storages
    # local
    shared_storage = get_shared_storage(storage_config=storage_config)
    replay_buffer = get_replay_buffer(storage_config=storage_config)

    # set start signal
    # wait until collecting enough data to start
    while not (replay_buffer.get_total_len() >= config.start_transitions):
        time.sleep(1)
        pass
    print(f'************* Begin Testing *************')
    shared_storage.set_start_signal()
