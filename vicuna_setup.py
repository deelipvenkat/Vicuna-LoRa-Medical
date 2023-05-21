from transformers import AutoTokenizer, LlamaConfig, LlamaTokenizer,LlamaForCausalLM, LlamaTokenizer,TextDataset,DataCollatorForLanguageModeling
from peft import get_peft_config, get_peft_model, PeftModel, PeftConfig ,LoraConfig,TaskType
from transformers import TrainingArguments, Trainer,GenerationConfig,LineByLineTextDataset
from datasets import Dataset , load_dataset
import torch
import accelerate
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
import textwrap
#import bitsandbytes
from torch.nn import CrossEntropyLoss
import numpy as np
import json
import pickle
from transformers import StoppingCriteria, StoppingCriteriaList
import re


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
    
    

stop_words_ids = [torch.tensor([835]).to('cuda'),torch.tensor([2277, 29937]).to('cuda')]  # '###' can be encoded in two different ways.
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

PROMPT_TEMPLATE = f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.
 
### Instruction:
[INSTRUCTION]
 
### Response:
"""

def create_prompt(instruction) :
    
    return PROMPT_TEMPLATE.replace("[INSTRUCTION]", instruction)



def generate_response(prompt, model,tokenizer,config) :
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to('cuda')
 
    generation_config = config

    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=stopping_criteria,
        )
    

def format_response(response,tokenizer) :
    decoded_output = tokenizer.decode(response.sequences[0])
    response = decoded_output.split("### Response:")[1].strip()
    return "\n".join(textwrap.wrap(response))


def vicuna_inference(prompt, model,tokenizer,config):
    """
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    """

    prompt = create_prompt(prompt)
    response = generate_response(prompt, model,tokenizer,config)
    b=format_response(response,tokenizer)
    a=b.split('###')
    return a[0]