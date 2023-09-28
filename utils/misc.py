import os
import psutil
import cv2
import random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from omegaconf import OmegaConf
from utils.typing import *
import subprocess

_timing_ = False

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42):
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def huggingface_online():
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["DIFFUSERS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"


def huggingface_offline():
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


def set_default_on_cuda():
    try:
        torch.set_default_device("cuda")
    except:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


def step_check(step, step_size, run_at_zero=False) -> bool:
    """Returns true based on current step and step interval. credit: nerfstudio"""
    if step_size == 0:
        return False
    return (run_at_zero or step != 0) and step % step_size == 0


def tic():
    import time

    if "_timing_" not in globals() or not _timing_:
        return

    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc(name=""):
    import time

    if "_timing_" not in globals() or not _timing_:
        return

    if "startTime_for_tictoc" in globals():
        print("$" * 30)
        print(
            f"{name} Elapsed time is "
            + str(time.time() - startTime_for_tictoc)
            + " seconds."
        )
        print("$" * 30)
        # logger.info(
        #     f"{name} Elapsed time is "
        #     + str(time.time() - startTime_for_tictoc)
        #     + " seconds."
        # )
    else:
        print("Toc: start time not set")


class Timer:
    def __init__(self):
        self.t = 0.0
        self.n = 0

    def tic(self):
        self.t = torch.cuda.Event(enable_timing=True)
        self.t.record()

    def toc(self):
        torch.cuda.synchronize()
        self.t = self.t.elapsed_time()
        self.n += 1

    def avg(self):
        return self.t / self.n


def save_fig(img, filename):
    tmp = Path("tmp")
    if not tmp.exists():
        tmp.mkdir()
    if not isinstance(img, np.ndarray):
        assert isinstance(img, torch.Tensor)
        img = img.detach()
        if img.size(0) == 1:
            img = img.squeeze()
        if img.size(0) == 3:
            img = img.moveaxs(0, -1)

        img = img.cpu().numpy()
    else:
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)

    fig, ax = plt.subplots()
    ax.imshow(img)
    fig.savefig(f"tmp/{filename}.png", bbox_inches="tight")


def save_img(img, path, filename):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)

    img = (img.detach().cpu().numpy() * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path / filename), img)


def float2uint8(img):
    if isinstance(img, torch.Tensor):
        img = (img * 255.0).to(torch.uint8)
    elif isinstance(img, np.ndarray):
        img = (img * 255.0).astype(np.uint8)
    else:
        raise NotImplementedError

    return img


def print_info(var, name):
    print(f"++++++{name}++++++")
    print("contiguous: ", var.is_contiguous())
    print(var.max())
    print(var.min())
    print(var.shape)
    if var.dtype == torch.float32:
        print("mean:", var.mean().item())
    print("nonzero:", torch.count_nonzero(var).item())
    if var.isnan().any():
        print("num nans", torch.count_nonzero(var.isnan()).item())
    print(f"++++++{name}++++++")


def mp_read(filenames):
    pass


def lineprofiler(func):
    try:
        func = profile(func)
    except NameError:
        pass
    return func


def average_dicts(dicts):
    avg_dict = {}
    for k in dicts[0].keys():
        avg_dict[k] = np.mean([d[k] for d in dicts])
    return avg_dict


def stack_dicts(dicts):
    stacked_dict = {}
    for k in dicts[0].keys():
        stacked_dict[k] = torch.stack([d[k] for d in dicts], dim=0)
    return stacked_dict


def reduce_dicts(dicts, fn):
    avg_dict = {}
    for k in dicts[0].keys():
        avg_dict[k] = fn(torch.stack([d[k] for d in dicts], dim=0), dim=0)
    return avg_dict


def sum_dicts(dicts):
    sum_dict = {}
    for k in dicts[0].keys():
        sum_dict[k] = torch.sum([d[k] for d in dicts])


def to_primitive(cfg, resolve=True):
    # convert omegaconf to primitive types, avoid errors in calls of type(cfg) and isinstance(cfg, ...)
    if (
        isinstance(cfg, float)
        or isinstance(cfg, int)
        or isinstance(cfg, str)
        or isinstance(cfg, list)
        or isinstance(cfg, dict)
    ):
        return cfg
    return OmegaConf.to_container(cfg, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def C(value: Any, step, max_steps=None) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        if len(value) == 4:
            assert len(value) == 4
            start_step, start_value, end_value, end_step = value
            if isinstance(end_step, int):
                current_step = step
                value = start_value + (end_value - start_value) * max(
                    min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
                )
            elif isinstance(end_step, float):
                if max_steps is None:
                    raise ValueError(
                        "max_steps must be specified when using float step, which mean the end of interpolation is int(max_steps * end_step)"
                    )
                current_step = end_step * max_steps
                value = start_value + (end_value - start_value) * max(
                    min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
                )
        else:
            assert len(value) == 5
            start_step, start_value, end_value, end_step, interp_type = value
            if interp_type == "linear":
                return C(
                    [start_step, start_value, end_value, end_step], step, max_steps
                )
            elif interp_type == "sqrt":
                current_step = step
                w = np.sqrt(
                    max(
                        min(1.0, (current_step - start_step) / (end_step - start_step)),
                        0.0,
                    )
                )
                t = end_value - (end_value - start_value) * w
                return t
            elif interp_type == "alternative":
                current_step = step
                if ((current_step - start_step) // (end_step - start_step)) % 2 == 0:
                    return start_value
                else:
                    return end_value
    return value


def C_wrapped(value, max_steps=None):
    def fn(s):
        return C(value, s, max_steps)

    return fn


def get_file_list():
    return [
        b.decode()
        for b in set(
            subprocess.check_output(
                'git ls-files -- ":!:load/*"', shell=True
            ).splitlines()
        )
        | set(  # hard code, TODO: use config to exclude folders or files
            subprocess.check_output(
                "git ls-files --others --exclude-standard", shell=True
            ).splitlines()
        )
    ]


def dict_to_device(dic, device):
    for key in dic.keys():
        if isinstance(dic[key], torch.Tensor):
            dic[key] = dic[key].to(device)

    return dic


def list_to_float(input):
    for i in range(len(input)):
        try:
            input[i] = float(input[i])
        except:
            pass
    return input


def load_as_pcd(ckpt):
    params = torch.load(ckpt, map_location="cpu")["params"]
    xyz = params["mean"]
    rgb = torch.sigmoid(params["color"])
    return torch.cat([xyz, rgb], dim=-1)


def get_current_cmd():
    my_process = psutil.Process(os.getpid())
    cmds = my_process.cmdline()
    cmd = " ".join(cmds)

    return cmd


def get_dict_slice(data, start, end):
    output = {}
    for key, val in data.items():
        try:
            output[key] = val[start:end]
        except:
            raise ValueError(f"Cannot slice {key} from {start} to {end}")

    return output


def get_ckpt_path(ckpt):
    ckpt = Path(ckpt)
    if not ckpt.exists():
        uid, time, day, prompt = str(ckpt).strip().split("|")

        ckpt_dir = Path(f"./checkpoints/{prompt}/{day}/{time}/ckpts/")
        files = ckpt_dir.glob("*.pt")
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        ckpt = latest_file

    return ckpt
