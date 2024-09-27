import os
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset 
dataset = load_dataset("M-A-D/Mixed-Arabic-Datasets-Repo", "Ara--Wikipedia", split="train")

HF_token = "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"
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Import Llama Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token= HF_token)

# Generate text batches
def get_batches(batch_size=1000):
  for i in range(0, len(dataset), batch_size):
    yield dataset[i:i+batch_size]["text"]

# Retrain Llama Tokenizer    
tokenizer = tokenizer.train_new_from_iterator(
    get_batches(),
    vocab_size = 32208,
    show_progress = True)

# Save the new tokenizer
tokenizer.save_pretrained("tokenizer")
