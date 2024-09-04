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

model.load_state_dict(torch.load("checkpoint.pth", map_location=device, weights_only=True))

text = decode(model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=10000)[0].tolist())

with open("generation.txt", "w") as f:
    f.write(text)

