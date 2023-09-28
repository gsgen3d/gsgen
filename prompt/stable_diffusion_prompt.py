import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline

from utils.typing import *
from utils.ops import perpendicular_component
from utils.misc import C
from .prompt_processors import BasePromptProcessor, PromptEmbedding


class StableDiffusionPromptProcessor(BasePromptProcessor):
    def prepare_text_encoder(self, guidance_model=None):
        if guidance_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="tokenizer",
                cache_dir="./.cache",
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                device_map="auto",
                cache_dir="./.cache",
            )
        else:
            self.tokenizer = guidance_model.pipe.tokenizer
            self.text_encoder = guidance_model.pipe.text_encoder

    def encode_prompts(self, prompts):
        with torch.no_grad():
            print(prompts)
            tokens = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).to(self.device)
            # print(tokens.input_ids.device)
            text_embeddings = self.text_encoder(tokens.input_ids)[0]

        return text_embeddings

    def update(self, step):
        pass
