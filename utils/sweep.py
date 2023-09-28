import argparse
import os
import nvitop
import time
import sys
import datetime
from nvitop import select_devices
from hydra import compose, initialize
from omegaconf import OmegaConf
from utils.misc import to_primitive
from pathlib import Path
from itertools import product
from rich.console import Console
from copy import deepcopy

console = Console()


def find_available_gpus(minial_free_memory=7.0, devices=None):
    """find availables gpus given conditions

    Args:
        minial_free_memory (float, optional): mininal free VRAM of gpus, in GiB. Defaults to 7.0.
    """
    if devices is not None:
        devices = [nvitop.Device(i) for i in devices]
    return select_devices(
        devices=devices, format="index", min_free_memory=f"{minial_free_memory}GiB"
    )


def find_gpus_without_process_of_current_user():
    n_devices = nvitop.Device.count()
    gpus = [nvitop.Device(i) for i in range(n_devices)]
    available_gpus = []
    user_name = os.environ["USER"]
    for gpu in gpus:
        no_current_user = True
        for process in gpu.processes().values():
            if process.username() == user_name:
                no_current_user = False
                break
        if no_current_user:
            available_gpus.append(gpu.index)

    return available_gpus


def set_cfg_field(cfg, field, value):
    fields = field.split(".")
    for f in fields[:-1]:
        cfg = cfg[f]
    cfg[fields[-1]] = value
    # return cfg


def generate_sweep_configs(base_name, sweep_config, sweep_group_name=None, base=0):
    initialize(config_path="../conf")
    cfg = compose(config_name=base_name)

    sweep_cfg = OmegaConf.load(f"./conf/sweep/{sweep_config}.yaml")

    joint_fields = to_primitive(sweep_cfg.joint_fields)
    joint_len = 0
    for field in joint_fields:
        if joint_len == 0:
            joint_len = len(sweep_cfg[field])
        else:
            assert joint_len == len(
                sweep_cfg[field]
            ), f"joint fields {field} have different length"

    if sweep_group_name is None:
        sweep_group_name = datetime.datetime.now().strftime("%Y-%m-%d")
    sweep_config_path = Path("./sweep") / sweep_group_name
    if not sweep_config_path.exists():
        sweep_config_path.mkdir(parents=True, exist_ok=True)

    all_fields = sweep_cfg.keys()
    cross_fields = list(filter(lambda x: x not in joint_fields, all_fields))
    cross_fields.remove("joint_fields")

    num_total_cfgs = (
        joint_len * len(list(product(*[sweep_cfg[f] for f in cross_fields])))
        if joint_len > 0
        else len(list(product(*[sweep_cfg[f] for f in cross_fields])))
    )
    console.print(
        f"Sweep {sweep_group_name} total number of configs: {num_total_cfgs}, joint fields: {joint_fields}, cross fields: {cross_fields}"
    )

    cnt = 0
    for cross_items in product(*[sweep_cfg[f] for f in cross_fields]):
        for c_idx, cross_item in enumerate(cross_items):
            set_cfg_field(cfg, cross_fields[c_idx], cross_item)

        if joint_len > 0:
            for joint_items in zip(*[sweep_cfg[f] for f in joint_fields]):
                for j_idx, joint_item in enumerate(joint_items):
                    set_cfg_field(cfg, joint_fields[j_idx], joint_item)

                cfg_name = f"{cnt + base}.yaml"
                OmegaConf.save(cfg, sweep_config_path / cfg_name)
                cnt += 1
        else:
            cfg_name = f"{cnt + base}.yaml"
            OmegaConf.save(cfg, sweep_config_path / cfg_name)
            cnt += 1

    assert (
        cnt == num_total_cfgs
    ), "number of configs does not match, this script is buggy"


if __name__ == "__main__":
    print(find_gpus_without_process_of_current_user())
    print(find_available_gpus(1.0, find_gpus_without_process_of_current_user()))
    parser = argparse.ArgumentParser()
    parser.add_argument("base_name", type=str)
    parser.add_argument("--sweep_config", type=str, default=None)
    parser.add_argument("--sweep_group_name", type=str, default=None)
    parser.add_argument("--offset", type=int, default=0)

    opt = parser.parse_args()

    if opt.sweep_config is not None:
        generate_sweep_configs(
            opt.base_name, opt.sweep_config, opt.sweep_group_name, opt.offset
        )
