import os
import numpy as np
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Load the dataset
dataset = load_dataset("M-A-D/Mixed-Arabic-Datasets-Repo", "Ara--Wikipedia", split="train")

# Split the dataset into training and validation sets
split_dataset = dataset.train_test_split(test_size=0.003, seed=2357, shuffle=False)
split_dataset['val'] = split_dataset.pop('test')

# Load the tokenizer from the JSON file
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

# Function to process each example
def process(example):
    tokens_ids = tokenizer.encode(example["text"])
    tokens_ids.append(tokenizer.eos_token_id)
    return {"token_ids": tokens_ids, "len": len(tokens_ids)}

# Process the dataset
new_dataset = split_dataset.map(
    process,
    num_proc=8,
    desc="Tokenizing the dataset",
    remove_columns=["id", "url", "title", "text"]
)

# Save the token IDs using memory mapping
for split, dset in new_dataset.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    print(f"Number of tokens in {split}: ", arr_len)
    filename = f'{split}.bin'
    dtype = np.uint16  
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    total_batches = 60
    idx = 0

    for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['token_ids'])
        
        arr[idx:idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)

    arr.flush()  
    del arr  

print("Token IDs saved to memory-mapped files.")
