import torch
import pickle
from model import MiniGPTConfig, MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("tokenizer.pkl", "rb") as f:
    tokenizer= pickle.load(f)

itos, stoi = tokenizer["itos"], tokenizer["stoi"]

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

config = MiniGPTConfig()
model = MiniGPT(config).to(device)

model_dict = torch.load("checkpoints/step_8000.pt", map_location=torch.device('cpu'))
def remove_prefix(state_dict):
    return {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}

model_dict = remove_prefix(model_dict["model"])
model.load_state_dict(model_dict)

model.eval()
tokens = encode("هي قصة")
text = decode(model.generate(torch.tensor(tokens, dtype=torch.long ,device=device).unsqueeze(0), max_new_tokens=1000)[0].tolist())

