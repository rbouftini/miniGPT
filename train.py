import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MiniGPTConfig, MiniGPT
import math

torch.manual_seed(1337)

dataset = "darija_stories.txt"
context_length = 512
batch_size = 64
n_embed = 384  #Number of embedding dimensions
n_layers = 8
n_heads = 8
dropout = 0.2
eval_iter = 20
max_iters = 5001
warmup_steps = 500
max_lr = 3e-4
min_lr = 3e-5
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    else:
      # Cosine decay down to min learning rate
      decay_ratio = (it - warmup_steps) / (max_iters - warmup_steps)
      assert 0 <= decay_ratio <= 1
      coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
      return min_lr + coeff * (max_lr - min_lr)


data_dir = os.path.join("data", dataset)
with open(data_dir, "r", encoding="utf-8") as f:
  text = f.read()

#Building the vocabulary
vocab = sorted(list(set(text)))
vocab_size = 512

model_parameters = dict( vocab_size = vocab_size, context_length = context_length, n_embed = n_embed, n_layers = n_layers,
    n_heads = n_heads, dropout = dropout)

#Tokenizing the text ~ Converting it to a sequence of integers according to our vocabulary
#Creating two dictionaries: one to represent a token into an unique integer
#Second to map back from the integer to the word
itos = {index:val for index,val in enumerate(vocab)}
stoi = {val:index for index,val in enumerate(vocab)}

tokenizer = {"itos": itos, "stoi" :stoi}
with open("tokenizer.pkl", "wb") as f:
  pickle.dump(tokenizer, f)

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype= torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Training data has: {len(train_data)} tokens")
print(f"Validation data has: {len(val_data)} tokens")

def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data)- context_length, (batch_size,))
  x = torch.stack([data[i:i+context_length]for i in ix])
  y = torch.stack([data[i+1:i+1+context_length]for i in ix])
  x,y = x.to(device), y.to(device)
  return x,y

config = MiniGPTConfig(**model_parameters)
model = MiniGPT(config).to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9,0.95), eps=1e-8, fused=True)
scaler = torch.amp.GradScaler(device='cuda')

@torch.no_grad()
def get_validation_loss():
  model.eval()
  total_loss = 0.0
  for iter in range(eval_iter):
    xb, yb = get_batch("val")
    with torch.autocast(device_type=device, dtype=torch.float16):
      _ , loss = model(xb,yb)
    total_loss += loss.item()
  total_loss /= eval_iter
  model.train()
  return total_loss

import time

for step in range(max_iters):
    step_start_time = time.perf_counter()
    xb, yb = get_batch("train")
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(xb, yb)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # Zeroing out gradients from the previous step
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    # Gradient clipping
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    step_end_time = time.perf_counter()
    step_time = step_end_time - step_start_time 
    # Print step time and losses periodically
    if step % 20 == 0:
        validation_loss = get_validation_loss()
        print(f"step {step}, loss: {validation_loss:.4f}, lr: {lr:.6e}, time: {step_time:.4f} seconds")

torch.save(model.state_dict(), 'checkpoint.pth')