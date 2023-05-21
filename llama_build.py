from transformers import AutoTokenizer,LlamaForCausalLM, LlamaTokenizer,TextDataset,DataCollatorForLanguageModeling
from peft import get_peft_model, PeftModel, PeftConfig ,LoraConfig,TaskType
from transformers import TrainingArguments, Trainer
from huggingface_hub import notebook_login
from datasets import Dataset , load_dataset

tokenizer = AutoTokenizer.from_pretrained("/home/vicuna-weights-7B")
print("starting model loading")
model = LlamaForCausalLM.from_pretrained("/home/vicuna-weights-7B")
print("model loaded")