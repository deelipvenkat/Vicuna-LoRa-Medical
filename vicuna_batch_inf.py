from transformers import AutoTokenizer, LlamaConfig, LlamaTokenizer,LlamaForCausalLM, LlamaTokenizer,TextDataset,DataCollatorForLanguageModeling
from peft import get_peft_model, PeftModel, PeftConfig ,LoraConfig,TaskType
from transformers import TrainingArguments, Trainer,GenerationConfig
from datasets import Dataset , load_dataset
import torch
import accelerate
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
import textwrap
#import bitsandbytes


# BATCH INFERENCE - TOKENIZER ISSUE

# for batch inference
PROMPT_TEMPLATE = f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.
 
### Instruction:
[INSTRUCTION]
 
### Response:
"""

def create_prompt(instruction) :
    return PROMPT_TEMPLATE.replace("[INSTRUCTION]", instruction)

def generate_batch_response(prompts, model,tokenizer,config):
    #tokenizer.pad_token_id = (
    #    0  
    #)
    #tokenizer.padding_side = "left"

    encodings = tokenizer.batch_encode_plus(prompts, return_tensors="pt", pad_to_max_length=True)
    input_ids = encodings["input_ids"].to('cuda')

    generation_config = config
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1  # Set the number of sequences to generate
        )


def format_response_batch(responses,tokenizer):
    formatted_responses = []
    decoded_output = tokenizer.batch_decode(responses.sequences)
    #print(decoded_output)
    for outputs in decoded_output:
        response_text = outputs.split("### Response:")[1].strip()
        response_text=response_text.split('###')[0]
        formatted_responses.append("\n".join(textwrap.wrap(response_text)))
    return formatted_responses

def format_response_batch_2(responses,tokenizer):
    formatted_responses = []

    
    outputs=tokenizer.batch_decode(responses)
    print(outputs)
    response_text = outputs.split("### Response:")[1].strip()
    response_text=response_text.split('###')[0]
    formatted_responses.append(response_text)
    return formatted_responses


def vicuna_batch_inf(prompts, model):
    prompt_list = [create_prompt(prompt) for prompt in prompts]
    responses = generate_batch_response(prompt_list, model)
    print(responses)
    formatted_responses = format_response_batch_2(responses)
    return formatted_responses


q_list=['hi there', 'what is batch inference','how are you ?', 'what is the weight of earth ?']