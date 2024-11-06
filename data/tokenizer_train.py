import argparse
import os 
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from dotenv import load_dotenv

def train_from_scratch():
    #Load HuggingFace API key
    load_dotenv() 
    HF_token = os.getenv('HF_TOKEN')

    # Load the dataset
    dataset = load_dataset('lightonai/ArabicWeb24', data_files='ArabicWeb24/*/.arrow', split='train', token=HF_token)
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"]) 

    # Import Llama Tokenizer
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token= HF_token)

    # Generate text batches
    def get_batches(batch_size=1000):
        batch = []
        for example in dataset:
            batch.append(example["text"])
            if len(batch) == batch_size:      
                yield batch  
                batch = []   
        if batch:  
            yield batch 
        
    # Retrain Llama Tokenizer
    tokenizer = tokenizer.train_new_from_iterator(  
        text_iterator = get_batches(),     
        vocab_size = 32208,  
        show_progress = True) 

    return tokenizer

def load_pretrained():
    # Load the Aranizer tokenizer
    tokenizer = AutoTokenizer.from_pretrained("riotu-lab/Aranizer-SP-64k")
    
    # Add special tokens similar to Llama 3.1
    special_tokens_dict = {
            "bos_token": "<begin_of_text>",
            "eos_token": "<end_of_text>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "additional_special_tokens": [
                "<start_header_id>",
                "<end_header_id>",
                "system",
                "user",
                "assistant"
            ]
    }
        
    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script trains a tokenizer from scratch or uses a pretrained one.",
        epilog="Example: python .\\data\\tokenizer_train.py --train_from_scratch"
    )
    parser.add_argument("--train_from_scratch", 
                        action="store_true",
                        help="Train a tokenizer from scratch on the data if set, otherwise use a pretrained tokenizer")
    args = parser.parse_args()
    if args.train_from_scratch:
        print("Training a new tokenizer from the data")
        tokenizer = train_from_scratch()
    else:
        print("Configuring pretrained tokenizer Aranizer-SP-64k")
        tokenizer = load_pretrained()

    # Save the tokenizer
    tokenizer.save_pretrained("tokenizer")
