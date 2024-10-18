#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import MllamaForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
import torch


# In[2]:


#model_id = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
model_id = 'Llama-3.2-11B-Vision-Instruct'


# Run only one of the following 2 cells...

# In[3]:


if True:
    # nf4 quant
    dtype = torch.bfloat16
    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type='nf4'
    )

    dest = 'Llama-3.2-11B-Vision-Instruct-bnb-nf4'


# In[ ]:


else:
    # LLM.int() quant
    dtype = torch.float16
    q_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=[
            # More of a "feel good" thing to skip quantizing embedding
            # & lm_head layers.
            'vision_model.patch_embedding',
            'vision_model.gated_positional_embedding',
            'vision_model.gated_positional_embedding.tile_embedding',
            'vision_model.pre_tile_positional_embedding',
            'vision_model.pre_tile_positional_embedding.embedding',
            'vision_model.post_tile_positional_embedding',
            'vision_model.post_tile_positional_embedding.embedding',
            'language_model.model.embed_tokens',
            'language_model.lm_head',
            # Quantizing the following leads to CUDA assertion errors during
            # inference, so skip it.
            'multi_modal_projector'
        ]
    )

    dest = 'Llama-3.2-11B-Vision-Instruct-bnb-int8'


# Unfortunately, you have to be able to load the entire quantized model into VRAM.
# 
# `nf4` will barely fit into 8 GB (but you won't be able to run inference on only 8!)
# 
# There's a way around this involving a custom `device_map` and setting `llm_int8_enable_fp32_cpu_offload=True` (yes, even on the `nf4` quant), but that would make the quantized model only loadable with that specific `device_map` and config.

# In[4]:


model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    # This doesn't set the dtype of the submodules (vision_model, language_model). Does it matter?
    torch_dtype=dtype,
    device_map='auto',
    quantization_config=q_config,
)


# The following should save
# 
# * `config.json`
# * `generation_config.json`
# 
# As well as the safetensors and `model.safetensors.index.json` file (if sharded)

# In[5]:


model.save_pretrained(dest)


# And the following should save:
# 
# * `chat_template.json`
# * `preprocessor_config.json`
# * `special_tokens_map.json`
# * `tokenizer.json`
# * `tokenizer_config.json`

# In[6]:


processor = AutoProcessor.from_pretrained(model_id)
processor.save_pretrained(dest)

