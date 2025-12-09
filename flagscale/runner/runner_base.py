import multiprocessing
import os
import shlex

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.utils import (
    get_free_port,
    get_host_name_or_ip,
    get_nnodes,
    get_nproc_per_node,
    logger,
    parse_hostfile,
    run_local_command,
    run_scp_command,
    run_ssh_command,
    update_cmd_with_node_specific_config,
    update_nodes_envs,
)


class JobStatus(Enum):
    RUNNING = "Running"
    TRANSITIONAL = "Transitional (Stopping or Starting)"
    COMPLETED_OR_IDLE = "Completed or Not Started"


class RunnerBase(ABC):
    def __init__(self, config: DictConfig):
        self.config = config
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type in [
            "train",
            "inference",
            "compress",
            "serve",
            "rl",
        ], f"Unsupported task type: {self.task_type}"
        hostfile = self.config.experiment.runner.get("hostfile", None)
        self.resources = parse_hostfile(hostfile) if hostfile else None

    @abstractmethod
    def generate_run_script(self, *args, **kwargs):
        raise NotImplementedError(
            f"Please implement generate_run_script method within {self.task_type} Runner"
        )

    @abstractmethod
    def generate_stop_script(self, host, node_rank):
        raise NotImplementedError(
            f"Please implement generate_stop_script method within {self.task_type} Runner"
        )

    def _run_each(
        self,
        host,
        master_addr,
        master_port,
        nnodes,
        node_rank,
        nproc_per_node,
        with_test=False,
        dryrun=False,
    ):
        export_cmd = []
        for k, v in self.user_envs.items():
            export_cmd += [f"{k}={v}"]

        cmd = shlex.join(export_cmd + ["python"] + [self.user_script] + self.user_args)

        logging_config = self.config.inference.logging
        host_run_script_file = self.generate_run_script(
            self.config, host, node_rank, cmd, background=True, with_test=with_test
        )

        if host != "localhost":
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            # Step 1: make sure the scripts_dir exists on the remote host
            run_ssh_command(host, f"mkdir -p {logging_config.scripts_dir}", ssh_port, dryrun)

            # Step 2: copy the host_run_script_file to the remote host
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                run_scp_command(
                    host, host_run_script_file, logging_config.scripts_dir, ssh_port, dryrun
                )

            # Step 3: run the host_run_script_file on the remote host
            run_ssh_command(host, f"bash {host_run_script_file}", ssh_port, dryrun)
        else:
            run_local_command(f"bash {host_run_script_file}", dryrun)

    def run(
        self,
        with_test=False,
        dryrun=False,
        monitor=False,
        interval=10,
        enable_log_collection=True,
        enable_diagnostic=True,
        enable_monitoring=False,
    ):
        num_visible_devices = None
        visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is not None and isinstance(visible_devices, str):
            visible_devices = visible_devices.split(",")
            num_visible_devices = len(visible_devices)

        runner_config = self.config.experiment.runner

        # If hostfile is provided, use the resources from the hostfile
        if self.resources is not None:
            nnodes_from_hostfile = len(self.resources.keys())
            nnodes_from_args = runner_config.get("nnodes", None)
            nnodes = get_nnodes(nnodes_from_hostfile, nnodes_from_args)
            available_ip = list(self.resources.keys())[0]
            available_port = get_free_port()
            for node_rank, (host, resource_info) in enumerate(self.resources.items()):
                if node_rank >= nnodes:
                    break
                nproc_from_hostfile = resource_info["slots"]
                nproc_from_args = runner_config.get("nproc_per_node", None)
                nproc_per_node = get_nproc_per_node(
                    nproc_from_hostfile, nproc_from_args, num_visible_devices
                )
                master_addr = runner_config.get("master_addr", available_ip)
                master_port = runner_config.get("master_port", available_port)
                self._run_each(
                    host,
                    master_addr,
                    master_port,
                    nnodes,
                    node_rank,
                    nproc_per_node,
                    with_test=with_test,
                    dryrun=dryrun,
                )
        else:
            # If hostfile is not provided, run the job on localhost
            nproc_from_args = runner_config.get("nproc_per_node", None)
            nproc_per_node = get_nproc_per_node(None, nproc_from_args, num_visible_devices)
            available_addr = runner_config.get("master_addr", "localhost")
            available_port = runner_config.get("master_port", get_free_port())
            self._run_each(
                "localhost",
                available_addr,
                available_port,
                1,
                0,
                nproc_per_node,
                with_test=with_test,
                dryrun=dryrun,
            )

    def _stop_each(self, host, node_rank):
        host_stop_script_file = self.generate_stop_script(host, node_rank)
        logging_config = self.config.inference.logging

        if host != "localhost":
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            # Step 1: make sure the scripts_dir exists on the remote host
            run_ssh_command(host, f"mkdir -p {logging_config.scripts_dir}", ssh_port)
            # Step 2: copy the host_run_script_file to the remote host
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                run_scp_command(host, host_stop_script_file, logging_config.scripts_dir, ssh_port)
            # Step 3: run the host_run_script_file on the remote host
            run_ssh_command(host, f"bash {host_stop_script_file}", ssh_port)
        else:
            run_local_command(f"bash {host_stop_script_file}")

    def stop(self):
        if self.resources is None:
            self._stop_each("localhost", 0)
            return

        nnodes = get_nnodes(len(self.resources), self.config.experiment.runner.get("nnodes", None))

        for node_rank, (host, _) in enumerate(self.resources.items()):
            if node_rank >= nnodes:
                break
            self._stop_each(host, node_rank)
