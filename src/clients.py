##########################################
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
##########################################
import pathlib
import argparse
import multiprocessing
import time
from multiprocessing import Process
multiprocessing.set_start_method('spawn', force=True)

def distribute_gpus(num_clients, gpu_ids=''):
	"""Assign only explicitly requested GPUs; an empty list selects CPU."""
	gpus = [gpu.strip() for gpu in gpu_ids.split(',') if gpu.strip()]
	if any(not gpu.isdigit() for gpu in gpus):
		raise ValueError("GPU IDs must be comma-separated non-negative integers")
	return [gpus[i % len(gpus)] if gpus else None for i in range(num_clients)]


def parse_bool(value):
	if str(value).lower() in ('true', '1', 'yes'):
		return True
	if str(value).lower() in ('false', '0', 'no'):
		return False
	raise ValueError("Expected true or false, got %r" % value)


def wait_for_clients(clients):
	try:
		while any(client.is_alive() for client in clients):
			failed = [client for client in clients if client.exitcode not in (None, 0)]
			if failed:
				raise RuntimeError("Client worker exited with status %s" % failed[0].exitcode)
			time.sleep(0.1)
		if any(client.exitcode != 0 for client in clients):
			raise RuntimeError("One or more client workers failed")
	finally:
		for client in clients:
			if client.is_alive():
				client.terminate()
		for client in clients:
			client.join(timeout=5)
			if client.is_alive():
				client.kill()
				client.join()


def main():

	parser = argparse.ArgumentParser(description="Flower Federated Learning - Clients")
	parser.add_argument("--server_address",     type=str,   default="127.0.0.1:10001",    required=False,  help="gRPC server address")
	parser.add_argument("--dataset_dir",        type=str,   default="../dataset",   required=False,  help="Relevant path to dataset")
	parser.add_argument("--gpu_ids", type=str, default="", help="comma-separated GPU IDs; empty uses CPU")
	parser.add_argument("--gpu_memory",         type=int,   default=1024,           required=False,  help="GPU Available Memory")
	parser.add_argument("--num_clients",        type=int,   default=2,              required=False,  help="Number of clients")
	parser.add_argument("--batch_size",         type=int,   default=64,             required=False,  help="Training Batch Size")
	parser.add_argument("--l_per",              type=float, default=0.1,            required=False,  help="Percentage of dataset to be used as labelled data")
	parser.add_argument("--u_per",              type=float, default=1.0,            required=False,  help="Percentage of dataset to be used as unlabelled data")
	parser.add_argument("--fedstar",            type=str,   default="False",        required=False,  help="Enabler for Semi-Supervised Learning")
	parser.add_argument("--pseudo_label",       type=str,   default="False",        required=False,  help="Enabler for Pseudo-labelling")
	parser.add_argument("--class_distribute",   type=str,   default="False",        required=False,  help="Enabler for distributing samples according to class")
	parser.add_argument("--seed",               type=int,   default=2021,           required=False,  help="Seed for clients data shuffling")
	parser.add_argument("--verbose",            type=int,   default=0,              required=False,  help="Train Detail to be displayed")
	args = parser.parse_args()

	if args.num_clients < 1 or args.batch_size < 1 or args.gpu_memory < 1:
		parser.error("num_clients, batch_size, and gpu_memory must be positive")
	clients_gpu = distribute_gpus(num_clients=args.num_clients, gpu_ids=args.gpu_ids)
	data_dir = str(pathlib.Path(args.dataset_dir).expanduser().resolve())

	# Load Configurations of Clients
	clients_data = [{'client_id': i, 'num_clients': args.num_clients, 'server_address': args.server_address,
		'dataset_dir': data_dir, 'batch_size': args.batch_size, 'gpu_id': clients_gpu[i], 
		'gpu_memory': args.gpu_memory, 'seed': args.seed, 'l_per': args.l_per, 'u_per': args.u_per, 
		'fedstar': parse_bool(args.fedstar), 'class_distribute': parse_bool(args.class_distribute), 'verbose': args.verbose,
		} for i in range(args.num_clients)]

	# Start Multi-processing Clients and wait for them to finish
	clients_queue = multiprocessing.Queue()
	clients = [Client(clients_queue) for i in range(args.num_clients)]
	[client.start() for client in clients]
	[clients_queue.put(client) for client in clients_data]
	[clients_queue.put(None) for i in range(args.num_clients)]
	wait_for_clients(clients)
	clients_queue.close()

class Client(Process):

	def __init__(self, queue):
		Process.__init__(self)
		self.queue = queue

	def run(self):
		while True:
			cfg = self.queue.get()
			if cfg is None:
				break
			# Configure GPU
			Client.setup_gpu(gpu=cfg["gpu_id"],gpu_memory=cfg["gpu_memory"])
			# Create Client
			from client import AudioClient
			client = AudioClient(client_id=cfg["client_id"], server_address=cfg["server_address"],
				num_clients=cfg["num_clients"], dataset_dir=cfg["dataset_dir"], l_per=cfg["l_per"], u_per=cfg["u_per"],
				batch_size=cfg["batch_size"], verbose=cfg["verbose"],  seed=cfg["seed"], 
				fedstar=cfg["fedstar"],
				class_distribute=cfg["class_distribute"])
			# Start Client
			client(introduce=bool(cfg["verbose"]))

	@staticmethod
	def setup_gpu(gpu, gpu_memory):
		os.environ["CUDA_VISIBLE_DEVICES"] = gpu if gpu is not None else ""
		import tensorflow as tf
		gpus = tf.config.list_physical_devices("GPU")
		if gpu is not None and not gpus:
			raise RuntimeError("Requested GPU %s is unavailable; use an empty --gpu_ids for CPU" % gpu)
		if gpus:
			tf.config.experimental.set_virtual_device_configuration(gpus[0],
				[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory)])

if __name__ == "__main__":
	multiprocessing.freeze_support()
	main()
	sys.exit(0)
