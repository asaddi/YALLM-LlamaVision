# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
import os
from typing import Any

from accelerate import cpu_offload_with_hook
from accelerate.utils import set_seed
from PIL.Image import Image
import torch
import torchvision.transforms.functional as F
from transformers import MllamaForConditionalGeneration, AutoProcessor

from .models import *

# Not sure how to turn accelerate off, so can't do manual management like the rest of ComfyUI
# from comfy.model_management import get_torch_device, unet_offload_device


MODELS = ModelManager()
MODELS.load()


ChatMessage = dict[str,str]
ChatHistory = list[ChatMessage]

SamplerSetting = tuple[str,Any]


class LlamaVisionModel:
    def __init__(self, model_id: str):
        # device = get_torch_device()
        # https://huggingface.co/docs/transformers/main/en/model_doc/mllama#transformers.MllamaForConditionalGeneration
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # TODO should this be configurable?
            device_map='auto', # Ugh... accelerate handles the model whether I specify this or not. Manual management not possible?!
        )
        # https://huggingface.co/docs/transformers/main/en/model_doc/mllama#transformers.MllamaProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.hook = None # Create this lazily

    def chat_completion(self, messages: ChatHistory, image: Image|None=None,
                        samplers: list[SamplerSetting]|None=None, seed: int|None=None,
                        offload: bool=True) -> str:
        if samplers is None:
            samplers = []

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors='pt',
        ).to(self.model.device)

        if seed is not None:
            # Constrain to 32-bits or else one of the RNGs will raise an
            # exception.
            set_seed(seed & 0xffff_ffff)

        gen_args = {}
        for k,v in samplers:
            # MllamaForConditionalGeneration apparently works with all of these
            # (or at least, does not complain)
            if k in ('temperature', 'top_p', 'top_k', 'min_p'):
                gen_args[k] = v
        # print(f'gen_args = {gen_args}')

        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        output = self.model.generate(
            **inputs,
            max_new_tokens=2048, # TODO This should be configurable, I think?
            **gen_args,
        ).to(torch.device('cpu'))

        prompt_len = inputs.input_ids.shape[-1]
        generated_ids = output[:, prompt_len:] # Grab only the new tokens
        result = self.processor.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if offload:
            self.offload()

        return result

    def offload(self):
        if self.hook is None:
            self.model, self.hook = cpu_offload_with_hook(self.model)
        else:
            self.hook.offload()


class LlamaVisionModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        MODELS.refresh()
        return {
            'required': {
                'model': (MODELS.CHOICES,)
            }
        }

    TITLE = 'LlamaVision Model'

    RETURN_TYPES = ('LLMMODEL',)
    RETURN_NAMES = ('llm_model',)

    FUNCTION = 'execute'

    CATEGORY = 'LlamaVision'

    def execute(self, model: str):
        model_path = MODELS.download(model)
        # print(f"Using model at {model_path}")
        llm = LlamaVisionModel(model_path)

        return (llm,)


class LLMSamplerSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'temperature': ('FLOAT', {
                    'min': 0.0,
                    'default': 0.6,
                }),
                'min_p': ('FLOAT', {
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.0,
                }),
                'top_p': ('FLOAT', {
                    'min': 0.0,
                    'max': 1.0,
                    'default': 0.9,
                }),
                'top_k': ('INT', {
                    'min': 0,
                    'default': 0,
                }),
            }
        }

    TITLE = 'LLM Sampler Settings'

    RETURN_TYPES = ('LLMSAMPLER',)
    RETURN_NAMES = ('llm_sampler',)

    FUNCTION = 'execute'

    CATEGORY = 'LlamaVision'

    def execute(self, temperature, min_p, top_p, top_k):
        samplers = []

        # NB Fixed order, as there doesn't seem to be a way to set order in
        # LlamaVision's generate function.
        # top_k -> top_p -> min_p -> temperature as this is llama.cpp's
        # default order, but I don't know if it makes sense...
        if top_k > 0:
            samplers.append(('top_k', top_k))
        if top_p < 1.0:
            samplers.append(('top_p', top_p))
        if min_p > 0.0:
            samplers.append(('min_p', min_p))
        samplers.append(('temperature', temperature))

        return (samplers,)


class LlamaVisionChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'llm_model': ('LLMMODEL',),
                'image': ('IMAGE',),
                'user_prompt': ('STRING', {
                    'multiline': True,
                }),
                'seed': ('INT', {
                    'min': 0,
                    'default': 0,
                    'max': 0xffffffff_ffffffff, # Internally, I know the seed is limited to 32-bits... should this be constrained too?
                }),
                # TODO bool for whether or not to keep model loaded?
            },
            'optional': {
                'llm_sampler': ('LLMSAMPLER',),
            },
        }

    TITLE = 'LlamaVision Chat'

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('completion',)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = 'execute'

    CATEGORY = 'LlamaVision'

    def execute(self, llm_model: LlamaVisionModel, user_prompt: str, image, seed: int, llm_sampler: list[SamplerSetting]|None=None):
        # Make sure it's not a chat-only LLM (*cough* like from ComfyUI-YALLM-node)
        if not hasattr(llm_model, 'processor'):
            raise RuntimeError(f'{LlamaVisionChat.TITLE} only works with {LlamaVisionModelNode.TITLE}!')

        image = image.permute(0, 3, 1, 2) # Convert to [B,C,H,W]

        messages = []
        # Note: Llama 3.2 Vision doesn't support system prompt when there's an image
        # if system_prompt:
        #     messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': user_prompt}
        ]})

         # Not sure if this is how we should deal with batched images, but
         # it seems to work? (NB: OUTPUT_IS_LIST is True above)
        result = []
        for img in image:
            pil_image = F.to_pil_image(img)
            completion = llm_model.chat_completion(messages, pil_image, samplers=llm_sampler, seed=seed, offload=False)
            result.append(completion)

        llm_model.offload()

        return (result,)


NODE_CLASS_MAPPINGS = {
    'LlamaVisionModel': LlamaVisionModelNode,
    'LlamaVisionChat': LlamaVisionChat,
    'LLMSamplerSettings': LLMSamplerSettings,
}
