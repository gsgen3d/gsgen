import torch
import torch.nn.functional as F

from transformers import T5EncoderModel, T5Tokenizer
from diffusers import DDIMScheduler, DDPMScheduler, IFPipeline

from utils.typing import *
from utils.ops import perpendicular_component
from utils.misc import C
from .prompt_processors import BasePromptProcessor, PromptEmbedding
from rich.console import Console

console = Console()


class DeepFloydPromptProcessor(BasePromptProcessor):
    def prepare_text_encoder(self, guidance_model=None):
        repeat_until_success = self.cfg.get("repeat_until_success", False)
        if repeat_until_success:
            success = False
            while not success:
                try:
                    self.text_encoder = T5EncoderModel.from_pretrained(
                        self.cfg.pretrained_model_name_or_path,
                        subfolder="text_encoder",
                        load_in_8bit=True,
                        variant="8bit",
                        device_map="auto",
                        cache_dir="./.cache",
                    )  # FIXME: behavior of auto device map in multi-GPU training
                    # self.pipe = IFPipeline.from_pretrained(
                    #     self.cfg.pretrained_model_name_or_path,
                    #     text_encoder=self.text_encoder,  # pass the previously instantiated 8bit text encoder
                    #     unet=None,
                    # )
                    self.tokenizer = T5Tokenizer.from_pretrained(
                        self.cfg.pretrained_model_name_or_path,
                        subfolder="tokenizer",
                        cache_dir="./.cache",
                    )
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    console.print(".", end="")
                else:
                    success = True
                    break
        else:
            self.text_encoder = T5EncoderModel.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="text_encoder",
                load_in_8bit=True,
                variant="8bit",
                device_map="auto",
                cache_dir="./.cache",
            )  # FIXME: behavior of auto device map in multi-GPU training
            # self.pipe = IFPipeline.from_pretrained(
            #     self.cfg.pretrained_model_name_or_path,
            #     text_encoder=self.text_encoder,  # pass the previously instantiated 8bit text encoder
            #     unet=None,
            # )
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                subfolder="tokenizer",
                cache_dir="./.cache",
            )

    def encode_prompts(self, prompts):
        with torch.no_grad():
            print(prompts)
            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=77,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            text_embeddings = self.text_encoder(
                text_input_ids.to(self.text_encoder.device),
                attention_mask=attention_mask.to(self.text_encoder.device),
            )
            text_embeddings = text_embeddings[0]

        # breakpoint()
        return text_embeddings

    def update(self, step):
        pass
