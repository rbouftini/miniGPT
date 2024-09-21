import os
import numpy as np
import re
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

# Load the dataset
dataset = load_dataset("alielfilali01/Darija-Stories-Dataset", split="train")

# Split the dataset into training and validation sets
split_dataset = dataset.train_test_split(test_size=0.01, seed=2357, shuffle=False)
split_dataset['val'] = split_dataset.pop('test')

# Load the tokenizer from the JSON file
tokenizer = Tokenizer.from_file("tokenizer.json")

# Function to remove emojis from text
def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" 
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+" 
    )
    return emoji_pattern.sub(r'', text)

# Function to process each example
def process(example):
    example = remove_emojis(example["Text"])
    tokens_ids = tokenizer.encode(example).ids
    return {"token_ids": tokens_ids, "len": len(tokens_ids)}

# Process the dataset
new_dataset = split_dataset.map(
    process,
    num_proc=8,
    desc="Tokenizing and cleaning the dataset",
    remove_columns=["ChapterName", "ChapterLink", "Author", "Text", "Tags"]
)

# Save the token IDs using memory mapping
for split, dset in new_dataset.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
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