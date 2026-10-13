"""Launch one local experiment and clean up only its own process groups."""
import argparse
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent


def build_commands(config_path):
    import yaml

    config_path = Path(config_path).expanduser().resolve()
    with config_path.open() as stream:
        cfg = yaml.safe_load(stream)
    global_cfg, server_cfg = cfg["global"], cfg["server"]
    data_cfg, model_cfg = cfg["dataset"], cfg["model"]
    dataset = Path(data_cfg["DATA_DIR"]).expanduser()
    if not dataset.is_absolute():
        dataset = config_path.parent / dataset
    address = str(global_cfg["SERVER_ADDRESS"])
    common = ["--server_address", address, "--dataset_dir", str(dataset.resolve()),
              "--num_clients", str(global_cfg["NUM_CLIENTS"]), "--seed", str(global_cfg["SEED"]),
              "--batch_size", str(model_cfg["BATCH_SIZE"])]
    server = [sys.executable, str(ROOT / "src/server.py"), *common,
              "--rounds", str(server_cfg["FEDERATED_ROUNDS"]),
              "--min_sample_size", str(server_cfg["NUM_CLIENTS_PER_ROUND"]),
              "--gpu_memory", str(server_cfg["GPU_MEMORY_SIZE"]),
              "--gpu_id", str(server_cfg.get("GPU_ID", "")),
              "--eval_step", str(server_cfg["EVALUATION_STEP"]),
              "--train_epochs", str(server_cfg["LOCAL_TRAIN_STEP"]),
              "--learning_rate", str(model_cfg["LR"])]
    clients = [sys.executable, str(ROOT / "src/clients.py"), *common,
               "--gpu_memory", str(cfg["clients"]["GPU_MEMORY_SIZE"]),
               "--gpu_ids", str(cfg["clients"].get("GPU_IDS", "")),
               "--l_per", str(data_cfg["LABELLED_PER"]), "--u_per", str(data_cfg["UNLABELED_PER"]),
               "--fedstar", str(global_cfg["FEDSTAR"]), "--class_distribute", str(global_cfg["C_DIST"])]
    if not 1 <= server_cfg["NUM_CLIENTS_PER_ROUND"] <= global_cfg["NUM_CLIENTS"]:
        raise ValueError("NUM_CLIENTS_PER_ROUND must be between 1 and NUM_CLIENTS")
    for key in ("FEDERATED_ROUNDS", "EVALUATION_STEP", "LOCAL_TRAIN_STEP"):
        if server_cfg[key] < 1:
            raise ValueError("%s must be positive" % key)
    if not 0 < data_cfg["LABELLED_PER"] <= 1 or not 0 <= data_cfg["UNLABELED_PER"] <= 1:
        raise ValueError("LABELLED_PER must be in (0, 1]; UNLABELED_PER must be in [0, 1]")
    if global_cfg["FEDSTAR"] and (data_cfg["LABELLED_PER"] == 1 or data_cfg["UNLABELED_PER"] == 0):
        raise ValueError("FedSTAR requires both labelled and unlabelled examples")
    return server, clients, address


def stop_process(process):
    # Each child is started in a new session; its workers share that session.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_local(server_command, client_command, address, startup_timeout=120):
    parsed = urlsplit("//" + address)
    host, port = parsed.hostname, parsed.port
    if host not in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "::") or port is None:
        raise ValueError("The launcher requires a local host:port address")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    # Refuse an occupied endpoint. Never terminate an existing listener.
    with socket.socket(family) as probe:
        probe.bind((host, port))
    connect_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    processes = []
    try:
        server = subprocess.Popen(server_command, cwd=ROOT, start_new_session=True)
        processes.append(server)
        deadline = time.monotonic() + startup_timeout
        while True:
            if server.poll() is not None:
                raise RuntimeError("Server exited before becoming ready (status %s)" % server.returncode)
            try:
                with socket.create_connection((connect_host, port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Server did not become ready within %.0f seconds" % startup_timeout)
                time.sleep(0.1)
        clients = subprocess.Popen(client_command, cwd=ROOT, start_new_session=True)
        processes.append(clients)
        while True:
            server_code, client_code = server.poll(), clients.poll()
            if client_code is not None and client_code != 0:
                return client_code
            if server_code is not None and server_code != 0:
                return server_code
            if client_code is not None:
                return server.wait(timeout=30)
            if server_code is not None:
                return clients.wait(timeout=30)
            time.sleep(0.1)
    finally:
        for process in reversed(processes):
            stop_process(process)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yml")
    parser.add_argument("--dry-run", action="store_true", help="show commands without starting processes")
    parser.add_argument("--startup-timeout", type=float, default=120)
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("the local process launcher requires a POSIX system")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    try:
        server, clients, address = build_commands(args.config)
        if args.dry_run:
            for command in (server, clients):
                print(" ".join(shlex.quote(value) for value in command))
            return 0
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        return run_local(server, clients, address, args.startup_timeout)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired, ImportError) as error:
        print("FedSTAR: %s" % error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
