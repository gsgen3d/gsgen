import os
import time
import hydra
from random import shuffle
from trainer import Trainer
from omegaconf import OmegaConf
import subprocess
from pathlib import Path
from rich.console import Console
import argparse
from utils.sweep import (
    find_available_gpus,
    find_gpus_without_process_of_current_user,
    generate_sweep_configs,
)

console = Console()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_group_name", type=str)
    parser.add_argument(
        "-m",
        "--minimal_free_memory",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "-t",
        "--delta_t",
        type=float,
        default=60,
        help="time interval between each sweep, in seconds",
    )
    parser.add_argument("--no_subprocess", action="store_true", help="run in os.system")
    parser.add_argument(
        "-b", "--base_name", type=str, default=None, help="base config name"
    )
    parser.add_argument(
        "-c",
        "--sweep_config",
        type=str,
        default=None,
        help="sweep config name, should be in conf/sweep",
    )
    parser.add_argument(
        "-g",
        "--sweep_group_name",
        type=str,
        default=None,
        help="sweep group name, will show in wandb",
    )

    opt = parser.parse_args()

    delta_t = opt.delta_t
    sweep_group_name = opt.sweep_group_name

    base_dir = Path(f"./sweep/{sweep_group_name}")
    output_dir = Path(f"./sweep_output/{sweep_group_name}")

    if not base_dir.exists():
        assert (
            opt.base_name is not None
            and opt.sweep_config is not None
            and opt.sweep_group_name is not None
        ), "base_name, sweep_config and sweep_group_name must be specified if sweep group does not exist, which will be used to call the sweep config generation script"
        generate_sweep_configs(opt.base_name, opt.sweep_config, opt.sweep_group_name)
        raise ValueError(
            f"{base_dir} does not exist, run `python utils/sweep.py` first (see README.md for details)"
        )
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    config_list = list(base_dir.iterdir())
    shuffle(config_list)
    procs = []

    try:
        while len(config_list) > 0:
            cfg_file = config_list.pop()
            cfg_name = cfg_file.stem
            cfg_path = cfg_file.parent

            available_gpus = find_available_gpus(
                opt.minimal_free_memory, find_gpus_without_process_of_current_user()
            )

            if len(available_gpus) > 0:
                gpu = available_gpus[0]
                console.print(f"[red bold]Running Task {cfg_name} on GPU {gpu}")
                output_file_dir = output_dir / cfg_name
                output_file_dir.mkdir(parents=True, exist_ok=True)
                stdout_file = output_file_dir / "stdout.txt"
                stderr_file = output_file_dir / "stderr.txt"
                stdout_file = stdout_file.open("w")
                stderr_file = stderr_file.open("w")

                if opt.no_subprocess:
                    os.system(
                        f"CUDA_VISIBLE_DEVICES={gpu} python main.py --config-path={str(cfg_path)} --config-name={cfg_name}"
                    )
                else:
                    sub_env = os.environ.copy()
                    sub_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                    process = subprocess.Popen(
                        [
                            "python",
                            "main.py",
                            f"--config-path={str(cfg_path)}",
                            f"--config-name={cfg_name}",
                            f"+group=sweep_{sweep_group_name}",
                            f"+tags=[sweep_{sweep_group_name}]",
                        ],
                        env=sub_env,
                        stdout=stdout_file,
                        stderr=stderr_file,
                    )
                    procs.append(process)
            else:
                config_list.append(cfg_file)

            time.sleep(delta_t)
        for proc in procs:
            proc.wait()
    except KeyboardInterrupt:
        if opt.no_subprocess:
            for proc in procs:
                proc.kill()
