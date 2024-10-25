# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
from enum import Enum, auto
import logging
from typing import Any

from accelerate import cpu_offload_with_hook
from PIL.Image import Image
import torch
import torchvision.transforms.functional as F
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaConfig,
    MllamaForConditionalGeneration,
    set_seed,
)

from .models import *

# Not sure how to turn accelerate off, so can't do manual management like the rest of ComfyUI
# from comfy.model_management import get_torch_device, unet_offload_device


logger = logging.getLogger('LlamaVision')


MODELS = ModelManager()
MODELS.load()


ChatMessage = dict[str,str]
ChatHistory = list[ChatMessage]

SamplerSetting = tuple[str,Any]


def has_image(messages: ChatHistory) -> bool:
    for message in messages:
        content = message['content']
        if not isinstance(content, str): # plain text messages can't contain images
            # check if any of the parts from a multi-modal message have an image
            return any([x for x in content if x['type'] == 'image'])
    return False


class Quant(Enum):
    DEFAULT = auto()
    NF4 = auto()
    INT8 = auto()


class LlamaVisionModel:
    def __init__(self, model_id: str, quant: Quant=Quant.DEFAULT):
        config = MllamaConfig.from_pretrained(model_id)

        model_quant = Quant.DEFAULT
        if (bnb_config := getattr(config, 'quantization_config', {})):
            if bnb_config.get('quant_method') == 'bitsandbytes':
                if bnb_config['load_in_4bit'] and bnb_config['bnb_4bit_quant_type'] == 'nf4':
                    # TODO Should I bother with FP4? Do people use that?
                    model_quant = Quant.NF4
                elif bnb_config['load_in_8bit']:
                    model_quant = Quant.INT8

        # Default to whatever the model wants, otherwise bfloat16
        dtype = config.torch_dtype
        if dtype is None:
            dtype = torch.bfloat16

        quantization_config = None
        # TODO Maybe don't bother if model_quant != Quant.DEFAULT?
        if quant == Quant.NF4:
            dtype = torch.bfloat16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quant == Quant.INT8:
            dtype = torch.float16
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=[
                    'vision_model.patch_embedding',
                    'vision_model.gated_positional_embedding',
                    'vision_model.gated_positional_embedding.tile_embedding',
                    'vision_model.pre_tile_positional_embedding',
                    'vision_model.pre_tile_positional_embedding.embedding',
                    'vision_model.post_tile_positional_embedding',
                    'vision_model.post_tile_positional_embedding.embedding',
                    'language_model.model.embed_tokens',
                    'language_model.lm_head',
                    # Apparently, it doesn't like this layer being quantized
                    'multi_modal_projector'
                    ],
            )

        logger.info(f'Loading model with dtype {dtype}')

        # https://huggingface.co/docs/transformers/main/en/model_doc/mllama#transformers.MllamaForConditionalGeneration
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map='auto',
            quantization_config=quantization_config,
        ).eval()

        # https://huggingface.co/docs/transformers/main/en/model_doc/mllama#transformers.MllamaProcessor
        self.processor = AutoProcessor.from_pretrained(model_id)

        if model_quant == Quant.DEFAULT and quant != Quant.DEFAULT:
            # Quantized on-the-fly
            self.quant = quant
        else:
            # If the model is quantized, that overrides everything else
            # It doesn't matter what we passed in via quantization_config
            self.quant = model_quant

        logger.info(f'Model loaded, quant = {self.quant}')

        self.hook = None # Create this lazily

    def chat_completion(self, messages: ChatHistory, image: Image|None=None,
                        samplers: list[SamplerSetting]|None=None, seed: int|None=None,
                        offload: bool=True) -> str:
        if samplers is None:
            samplers = []

        if image is not None and not has_image(messages):
            # Work backwards through history
            for msg in reversed(messages):
                # And grab the very last user message we see
                if msg.get('role') == 'user':
                    parts = msg.get('content', [])
                    if isinstance(parts, str):
                        parts = [{'type': 'text', 'text': parts}]

                    # And insert image placeholder just before the user prompt
                    parts.insert(0, { 'type': 'image' })

                    msg['content'] = parts
                    break

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

        with torch.no_grad():
            # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
            output = self.model.generate(
                **inputs,
                max_new_tokens=2048, # TODO This should be configurable, I think?
                **gen_args,
            )

        prompt_len = inputs.input_ids.shape[-1]
        generated_ids = output[:, prompt_len:] # Grab only the new tokens
        result = self.processor.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if offload:
            self.offload()

        return result

    def offload(self):
        if self.quant == Quant.INT8:
            # Apparently you can't offload an int8 quant.
            logger.info('Asked to offload model, but model is int8 -- ignoring')
            return

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
                'model': (MODELS.CHOICES,),
                'quantization': ([q.name.lower() for q in Quant],)
            }
        }

    TITLE = 'LlamaVision Model'

    RETURN_TYPES = ('LLMMODEL',)
    RETURN_NAMES = ('llm_model',)

    FUNCTION = 'execute'

    CATEGORY = 'LlamaVision'

    def execute(self, model: str, quantization: str):
        quant = Quant[quantization.upper()]

        model_path = MODELS.download(model)
        # print(f"Using model at {model_path}")
        llm = LlamaVisionModel(model_path, quant=quant)

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
