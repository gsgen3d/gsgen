import gc
import requests
from . import BaseGuidance
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    IFPipeline,
    PNDMScheduler,
    DiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available

from utils.typing import *
from utils.ops import perpendicular_component
from utils.misc import C
from rich.console import Console

console = Console()


# TODO: finish this
class Zero123Guidance(BaseGuidance):
    def __init__(self, cfg):
        super().__init__(cfg)
