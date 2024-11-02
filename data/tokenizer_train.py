import os
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from dotenv import load_dotenv

# Load the different datasets into a single merged one
datasets = [
    load_dataset("M-A-D/Mixed-Arabic-Datasets-Repo", name, split="train")
    .remove_columns([col for col in load_dataset("M-A-D/Mixed-Arabic-Datasets-Repo", name, split="train").column_names if col != column])
    .rename_column(column, "Text")
    .filter(lambda example: example["Text"] is not None)
    for name, column in [
        ("Ara--Ali-C137--Hindawi-Books-dataset", "ChapterText"),
        ("Ara--Wikipedia", "text"),
        ("Ara--J-Mourad--MNAD.v1", "Body"),
        ("Arz--Wikipedia", "text"),
        ("Ara--saudinewsnet", "content")
    ]
]

merged_dataset = concatenate_datasets(datasets)

#Load HuggingFace API key
load_dotenv()
HF_token = os.getenv('HF_TOKEN')

# Import Llama Tokenizer
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token= HF_token)

# Generate text batches
def get_batches(batch_size=1000):
  for i in range(0, len(merged_dataset), batch_size):
    yield merged_dataset[i:i+batch_size]["Text"]

# Retrain Llama Tokenizer
tokenizer = tokenizer.train_new_from_iterator(
    text_iterator = get_batches(),
    vocab_size = 32208,
    show_progress = True)

# Save the new tokenizer
tokenizer.save_pretrained("tokenizer")
