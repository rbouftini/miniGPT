import torch
import pickle
from transformers import AutoTokenizer
from model import MiniGPT, MiniGPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("tokenizer")

config = MiniGPTConfig()
model = MiniGPT(config).to(device)

model_dict = torch.load("checkpoints/step_4000.pt", map_location=device)
def remove_prefix(state_dict):
    return {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}

model_dict = remove_prefix(model_dict["model"])
model.load_state_dict(model_dict)

model.eval()
example = "مدينة سان فرانسيسكو تقع"
text = tokenizer.decode(model.generate(torch.tensor(tokenizer.encode(example), dtype=torch.long ,device=device).unsqueeze(0), max_new_tokens=100)[0].tolist())