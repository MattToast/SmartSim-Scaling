import os
import os.path as osp
import shutil
import sys
import configparser
import datetime
from pathlib import Path
from itertools import product
from tqdm import tqdm
from uuid import uuid4
import pandas as pd
from imagenet.model_saver import save_model
from smartsim.error.errors import *
from smartsim.wlm import detect_launcher
import configparser
import time


import smartsim
from smartsim import Experiment, status
from smartredis import Client


from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")

def get_date():
    """Return current date

    This function will return the current date as a string.

    :return: date str
    :rtype: str
    """
    date = str(datetime.datetime.now().strftime("%Y-%m-%d"))
    return date

def get_time():
    """Return current time

    This function will return the current time as a string.

    :return: current_time str
    :rtype: str
    """
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time

def check_model(device, force_rebuild=False):
    """Regenerate model on specified device if True.

    This function will rebuild the model on the specified node type.

    :param device: device used to run the models in the database
    :type device: str
    :param force_rebuild: force rebuild of PyTorch model even if it is available
    :type force_rebuild: bool
    """
    if device.startswith("GPU") and (force_rebuild or not Path("./imagenet/resnet50.GPU.pt").exists()):
        from torch.cuda import is_available
        if not is_available():
            message = "resnet50.GPU.pt model missing in ./imagenet directory. \n"
            message += "Since CUDA is not available on this node, the model cannot be created. \n"
            message += "Please run 'python imagenet/model_saver.py --device GPU' on a node with an available CUDA device."
            logger.error(message)
            sys.exit(1)

def create_experiment_and_dir(exp_name, launcher):
    """Create and generate Experiment as well as create results folder.

    This function is called for every scaling test. It creates an Experiment per scaling test
    as well as the results folder to store each run folder.

    :param exp_name: name of output dir
    :type exp_name: str
    :param launcher: workload manager i.e. "slurm", "pbs"
    :type launcher: str
    :return: Experiment instance
    :rtype: Experiment
    """
    result_path = osp.join("results", exp_name, "run-" + get_date()+ "-" + get_time()) #autoincrement
    os.makedirs(result_path)
    #Ask Bill if I need a try/except block here even tho exp.generate has a try/except
    try:
        exp = Experiment(name=result_path, launcher=launcher)
        exp.generate()
    except Exception as e:
        logger.error(e)
        raise
    
    log_to_file(f"{exp.exp_path}/scaling-{get_date()}.log")
    return exp, result_path

def start_database(exp, db_node_feature, port, nodes, cpus, tpq, net_ifname, run_as_batch, hosts, wall_time):
    """Create and start the Orchestrator

    This function is only called if the scaling tests are
    being executed in the standard deployment mode where
    separate computational resources are used for the app
    and the database.
    :param exp: Experiment object for this test
    :type exp: Experiment
    :param db_node_feature: dict of runsettings for the database
    :type db_node_feature: dict[str,str]
    :param port: port number of database
    :type port: int
    :param nodes: number of database nodes
    :type nodes: int
    :param cpus: number of cpus per node
    :type cpus: int
    :param tpq: number of threads per queue
    :type tpq: int
    :param net_ifname: network interface name
    :type net_ifname: str
    :param run_as_batch: run database as separate batch submission each iteration
    :type run_as_batch: bool
    :param hosts: host to use for the database
    :type hosts: int
    :param wall_time: allotted time for database launcher to run
    :type wall_time: str
    :return: orchestrator instance
    :rtype: Orchestrator
    """
    db = exp.create_database(port=port,
                            db_nodes=nodes,
                            batch=run_as_batch,
                            interface=net_ifname,
                            threads_per_queue=tpq,
                            single_cmd=True,
                            hosts=hosts)
    if run_as_batch:
        db.set_walltime(wall_time)
        for k, v in db_node_feature.items():
            db.set_batch_arg(k, v)
    db.set_cpus(cpus)
    exp.generate(db, overwrite=True)
    exp.start(db)
    logger.info("Orchestrator Database created and running")
    return db

def setup_resnet(model, device, num_devices, batch_size, address, cluster=True):
    """Set and configure the PyTorch resnet50 model for inference

    :param model: path to serialized resnet model
    :type model: str
    :param device: CPU or GPU
    :type device: str
    :param num_device: number of devices per compute node to use to run ResNet
    :type num_device: int
    :param batch_size: batch size for the Orchestrator (batches of batches)
    :type batch_size: int
    :param address: address of database
    :type address: str
    :param cluster: true if using a cluster orchestrator
    :type cluster: bool
    """
    client = Client(address=address, cluster=cluster)
    device = device.upper()
    if (device == "GPU") and (num_devices > 1):
        client.set_model_from_file_multigpu("resnet_model",
                                            model,
                                            "TORCH",
                                            0, num_devices,
                                            batch_size)
        client.set_script_from_file_multigpu("resnet_script",
                                             "./imagenet/data_processing_script.txt",
                                             0, num_devices)
        logger.info(f"Resnet Model and Script in Orchestrator on {num_devices} GPUs")
    else:
        # Redis does not accept CPU:<n>. We are either
        # setting (possibly multiple copies of) the model and script on CPU, or one
        # copy of them (resnet_model_0, resnet_script_0) on ONE GPU.
        client.set_model_from_file(f"resnet_model",
                                    model,
                                    "TORCH",
                                    device,
                                    batch_size)
        client.set_script_from_file(f"resnet_script",
                                    "./imagenet/data_processing_script.txt",
                                    device)
        logger.info(f"Resnet Model and Script in Orchestrator on device {device}")

def write_run_config(path, **kwargs):
    """Write config attributes to run file.

    This function will write the config attributes to the run folder.

    :param path: path to model
    :type path: str
    :param kwargs: config attributes
    :type kwargs: keyword arguments
    """
    name = os.path.basename(path)
    config = configparser.ConfigParser()
    config["run"] = {
        "name": name,
        "path": path,
        "smartsim_version": smartsim.__version__,
        "smartredis_version": "0.3.1", # TODO put in smartredis __version__
        "db": get_db_backend(),
        "date": str(get_date()),
        "language": kwargs['language']
    }
    config["attributes"] = kwargs

    config_path = Path(path) / "run.cfg"
    with open(config_path, "w") as f:
        config.write(f)

def get_uuid():
    """Return the uuid.

    :return: uid str
    :rtype: str
    """
    uid = str(uuid4())
    return uid[:4]
    

def get_db_backend():
    """Return database backend.

    :return: db backend name str
    :rtype: str
    """
    db_backend_path = smartsim._core.config.CONFIG.database_exe
    return os.path.basename(db_backend_path)

def check_node_allocation(client_nodes, db_nodes):
    """Check if a user has the correct node allocation on a machine.

    :param client_nodes: list of nodes
    :type client_nodes: list<int>
    :param db_nodes: list of db nodes
    :type db_nodes: list<int>
    """
    if not db_nodes:
        raise ValueError("db_nodes cannot be empty")
    if not client_nodes:
        raise ValueError("client_nodes cannot be empty")
    
    if detect_launcher() == "slurm":
        total_nodes = os.getenv("SLURM_NNODES")
    else:
        total_nodes = os.getenv("PBS_NNODES")
    for perm in list(product(client_nodes, db_nodes)):
        one, two = perm
        val = one + two
        if val > int(total_nodes):
            raise AllocationError(f"Addition of db_nodes and client_nodes is {val} nodes but you allocated only {total_nodes} nodes")

def print_yml_file(path, logger):
    """Print the YML file contents to terminal for user.

    :param path: path to yml file
    :type path: str
    :param logger: name of logger
    :type logger: str
    """
    config = configparser.ConfigParser()
    config.read(path)
    for key, value in config._sections['run'].items():
        logger.info(f"Running {key} with value: {value}")
    for key, value in config._sections['attributes'].items():
        logger.info(f"Running {key} with value: {value}")

def check_database_folder(result_path, logger):
    """Cleans the database folder within results.

    :param result_path: path to results folder
    :type result_path: str
    :param logger: name of logger
    :type logger: str
    """
    time.sleep(5)
    for _ in range(5):
        rdb_folders = os.listdir(Path(result_path) / "database")
        for fold in rdb_folders:
            if '.rdb' in fold:
                logger.debug(f"Database folder removed: {fold}")
                os.remove(Path(result_path) / "database" / fold)
                break
        time.sleep(1)