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
max_tokens = 100

def generate_completion(prompt):
    tokens_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    for token in prompt:
        yield token

    for _ in range(max_tokens):
        new_token_id = model.generate(tokens_ids, max_new_tokens=1)[:,-1].unsqueeze(0)
        tokens_ids = torch.cat((tokens_ids, new_token_id), dim=-1)
        output = tokenizer.decode(new_token_id[0], skip_special_tokens=True)
        yield output

        if new_token_id.item() == tokenizer.eos_token_id:
            break