from typing import Any

from comfy.model_management import get_torch_device, unet_offload_device
import torch
import torchvision.transforms.functional as F
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np


ChatMessage = dict[str,str]
ChatHistory = list[ChatMessage]

SamplerSetting = tuple[str,Any]


class LlamaVisionModel:
    def __init__(self, model_id: str):
        device = get_torch_device()
        # https://huggingface.co/docs/transformers/main/en/model_doc/mllama#transformers.MllamaForConditionalGeneration
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # TODO should this be configurable?
            device_map='auto', # Ugh... accelerate handles it whether I specify this or not. Manual management not possible?!
        )
        # https://huggingface.co/docs/transformers/main/en/model_doc/mllama#transformers.MllamaProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)

    def chat_completion(self, messages: ChatHistory, samplers: list[SamplerSetting]|None=None, seed: int|None=None) -> str:
        raise NotImplementedError('WIP')


class LlamaVisionModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ([
                    'H:\TEMP\models\Llama-3.2-11B-Vision-Instruct-nf4', # TODO
                ],)
            }
        }

    TITLE = 'LlamaVision Model'

    RETURN_TYPES = ('LLMMODEL',)
    RETURN_NAMES = ('llm_model',)

    FUNCTION = 'execute'

    CATEGORY = 'LlamaVision'

    def execute(self, model: str):
        llm = LlamaVisionModel(model)

        return (llm,)


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
                    'max': 0xffffffff_ffffffff, # TODO double check. Also need to figure out how to pass seed to generator...
                }),
                # TODO bool for whether or not to keep model loaded?
            },
            'optional': {
                'llm_sampler': ('LLMSAMPLER',),
                'system_prompt': ('STRING', {
                    'multiline': True,
                })
            },
        }

    TITLE = 'LlamaVision Chat'

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('completion',)

    FUNCTION = 'execute'

    CATEGORY = 'LlamaVision'

    def execute(self, llm_model: LlamaVisionModel, user_prompt: str, image, seed: int, llm_sampler: list[SamplerSetting]|None=None, system_prompt: str|None=None):
        # Make sure it's not a chat-only LLM (*cough* like from ComfyUI-YALLM-node)
        if not hasattr(llm_model, 'processor'):
            raise RuntimeError(f'{LlamaVisionChat.TITLE} only works with {LlamaVisionModelNode.TITLE}!')

        image = image.permute(0, 3, 1, 2)
        img = image[0] # FIXME batching?!
        pil_image = F.to_pil_image(img)

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': user_prompt}
        ]})

        # TODO move this stuff to LlamaVisionModel.chat_completion

        # device = get_torch_device()
        # llm_model.model.to(device)
        input_text = llm_model.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = llm_model.processor(
            pil_image, # FIXME I very much doubt this works?! will need to convert to tensor of right shape
            input_text,
            add_special_tokens=False,
            return_tensors='pt',
        ).to(llm_model.model.device)

        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        output = llm_model.model.generate(
            **inputs,
            max_new_tokens=2048,
        ).to(torch.device('cpu')) # TODO seed? samplers? should max tokens be configurable?
        # Note: seed not accepted.

        prompt_len = inputs.input_ids.shape[-1]
        generated_ids = output[:, prompt_len:] # Grab only the new tokens
        result = llm_model.processor.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # TODO make this a switch?
        # Also pretending like we're unet? Is that fine?
        # Other options: text encoder, vae. Do we have access to comfy.model_management's internals?
        # llm_model.model.to(unet_offload_device())

        return (result,)

NODE_CLASS_MAPPINGS = {
    'LlamaVisionModel': LlamaVisionModelNode,
    'LlamaVisionChat': LlamaVisionChat,
}
