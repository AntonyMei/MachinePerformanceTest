import os
import time
import SMOS
from multiprocessing.managers import BaseManager

from core.storage_config import StorageConfig


class WatchdogServer(object):
    def __init__(self):
        self.reanalyze_batch_count = 0
        self.training_step_count = 0

    def increase_reanalyze_batch_count(self):
        self.reanalyze_batch_count += 1

    def get_reanalyze_batch_count(self):
        return self.reanalyze_batch_count

    def increase_training_step_count(self):
        self.training_step_count += 1

    def get_training_step_count(self):
        return self.training_step_count


class WatchdogServerManager(BaseManager):
    pass


def start_watchdog_server(storage_config: StorageConfig):
    """
    Start a watchdog server. Call this method remotely.
    """
    # initialize watchdog server
    watchdog_server = WatchdogServer()
    WatchdogServerManager.register('get_watchdog_server', callable=lambda: watchdog_server)

    # start server
    watchdog_connection = storage_config.watchdog_server_connection
    manager = WatchdogServerManager(address=(watchdog_connection.ip,
                                             watchdog_connection.port),
                                    authkey=bytes(watchdog_connection.authkey))
    server = manager.get_server()
    server.serve_forever()


def get_watchdog_server(storage_config: StorageConfig):
    """
    Get connection to a watchdog server.
    """
    WatchdogServerManager.register('get_watchdog_server')
    watchdog_server_connection = storage_config.watchdog_server_connection
    watchdog_server_manager = WatchdogServerManager(address=(watchdog_server_connection.ip,
                                                             watchdog_server_connection.port),
                                                    authkey=watchdog_server_connection.authkey)
    watchdog_server_connected = False
    while not watchdog_server_connected:
        try:
            watchdog_server_manager.connect()
            watchdog_server_connected = True
        except ConnectionRefusedError:
            time.sleep(1)
    watchdog_server = watchdog_server_manager.get_watchdog_server()
    return watchdog_server


def start_watchdog(storage_config: StorageConfig, mode=None):
    """
    Start a watchdog that monitors training statistics. Call this method remotely.
    mode: trainer / worker
    """
    # get watchdog server
    WatchdogServerManager.register('get_watchdog_server')
    watchdog_server_connection = storage_config.watchdog_server_connection
    watchdog_server_manager = WatchdogServerManager(address=(watchdog_server_connection.ip,
                                                             watchdog_server_connection.port),
                                                    authkey=bytes(watchdog_server_connection.authkey))
    watchdog_server_connected = False
    while not watchdog_server_connected:
        try:
            watchdog_server_manager.connect()
            watchdog_server_connected = True
        except ConnectionRefusedError:
            time.sleep(1)
    watchdog_server = watchdog_server_manager.get_watchdog_server()

    # get SMOS client
    smos_client = SMOS.Client(connection=storage_config.smos_connection)
    while True:
        time.sleep(10)
        if mode == "worker":
            _, replay_buffer_size = smos_client.get_entry_count(name=storage_config.replay_buffer_name)
