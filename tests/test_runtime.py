"""Launcher checks use local dummy processes, never the audio training runtime."""
import importlib.util
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_launcher():
    spec = importlib.util.spec_from_file_location("fedstar_run", ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_clients():
    spec = importlib.util.spec_from_file_location("fedstar_clients", ROOT / "src/clients.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LauncherTests(unittest.TestCase):
    def test_paths_are_config_relative_and_addresses_are_intact(self):
        run = load_launcher()
        server, clients, address = run.build_commands(ROOT / "config.yml")
        self.assertEqual(server[0], sys.executable)
        self.assertTrue(Path(server[1]).is_file())
        self.assertTrue(Path(clients[1]).is_file())
        for command in (server, clients):
            dataset = command[command.index("--dataset_dir") + 1]
            self.assertEqual(Path(dataset), ROOT / "datasets/speech_commands")
            self.assertEqual(command[command.index("--server_address") + 1], address)
        self.assertEqual(address, "127.0.0.1:10001")

    def test_occupied_port_preserves_existing_listener(self):
        run = load_launcher()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            address = "127.0.0.1:%d" % listener.getsockname()[1]
            with self.assertRaises(OSError):
                run.run_local([sys.executable, "-c", "raise AssertionError('must not start')"], [], address)
            with socket.create_connection(listener.getsockname(), timeout=1):
                pass

    @unittest.skipUnless(os.name == "posix", "process-group cleanup uses POSIX")
    def test_client_failure_propagates_and_cleans_owned_server(self):
        run = load_launcher()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "server.pid"
            server = [sys.executable, "-c", (
                "import os,pathlib,socket,time; "
                "pathlib.Path(%r).write_text(str(os.getpid())); "
                "s=socket.socket(); s.bind(('127.0.0.1',%d)); s.listen(); time.sleep(20)"
            ) % (str(pidfile), port)]
            started = time.monotonic()
            result = run.run_local(server, [sys.executable, "-c", "raise SystemExit(7)"],
                                   "127.0.0.1:%d" % port, startup_timeout=3)
            self.assertEqual(result, 7)
            self.assertLess(time.monotonic() - started, 6)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pidfile.read_text()), 0)

    def test_failed_server_does_not_start_clients(self):
        run = load_launcher()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "client-started"
            client = [sys.executable, "-c", "from pathlib import Path; Path(%r).touch()" % str(marker)]
            with self.assertRaises(RuntimeError):
                run.run_local([sys.executable, "-c", "raise SystemExit(9)"], client,
                              "127.0.0.1:%d" % port, startup_timeout=2)
            self.assertFalse(marker.exists())


class ClientProcessTests(unittest.TestCase):
    def test_explicit_cpu_and_round_robin_gpu_assignment(self):
        clients = load_clients()
        self.assertEqual(clients.distribute_gpus(3, gpu_ids=""), [None, None, None])
        self.assertEqual(clients.distribute_gpus(3, gpu_ids="2,3"), ["2", "3", "2"])

    def test_worker_failure_stops_remaining_owned_workers(self):
        clients = load_clients()
        ctx = multiprocessing.get_context("spawn")
        workers = [ctx.Process(target=sys.exit, args=(6,)), ctx.Process(target=time.sleep, args=(20,))]
        for worker in workers:
            worker.start()
        try:
            with self.assertRaises(RuntimeError):
                clients.wait_for_clients(workers)
            self.assertTrue(all(not worker.is_alive() for worker in workers))
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                worker.join()


if __name__ == "__main__":
    unittest.main()
