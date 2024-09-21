from datasets import load_dataset
import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Load the dataset
dataset = load_dataset("alielfilali01/Darija-Stories-Dataset", split="train")
text = ''.join([text for text in dataset["Text"]])

# Remove emojis
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

clean_text = remove_emojis(text)

# Initialize a tokenizer
tokenizer = Tokenizer(BPE())

# Initialize a trainer with special tokens
trainer = BpeTrainer(
    special_tokens=["<|endoftext|>"],
    vocab_size=16384  
)

# Configure pre-tokenization
tokenizer.pre_tokenizer = Whitespace()

# Prepare the text for training
def batch_iterator(batch_size=1000):
    for i in range(0, len(clean_text), batch_size):
        yield clean_text[i:i+batch_size]

# Train the tokenizer
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(clean_text))

# Save the tokenizer
tokenizer.save("tokenizer.json")

print("Tokenizer trained and saved succesfully")
