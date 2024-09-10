from datasets import load_dataset
import re
import os 

dataset = load_dataset("alielfilali01/Darija-Stories-Dataset", split="train")

text = ''.join([text for text in dataset["Text"]])

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

with open(os.path.join(os.path.dirname(__file__), "darija_stories.txt"), "w", encoding="utf-8") as f:
  f.write(clean_text)