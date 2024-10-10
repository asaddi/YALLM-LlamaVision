from transformers import MllamaForConditionalGeneration, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
from accelerate import Accelerator

model_id = r"Llama-3.2-11B-Vision-Instruct"

q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type='nf4')

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    quantization_config=q_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dest = 'Llama-3.2-11B-Vision-Instruct-nf4'

# AFAIK, the model needs to reside on a single device before you save it
# However, moving it gets you a complaint from the accelerate module
# So I do it in 2 steps here... First the config:
model.config.save_pretrained(dest)

# Then the model (I assume accelerate knows how to handle multi-device models)
accelerator = Accelerator()
accelerator.save_model(model, dest)

# We'll just copy the preprocessor & tokenizer files manually. Is there a better way to do it?
# PS Don't forget chat_template.json
