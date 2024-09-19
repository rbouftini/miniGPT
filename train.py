import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MiniGPTConfig, MiniGPT
import math
import time

torch.manual_seed(1337)

dataset = "darija_stories.txt"
context_length = 512
batch_size = 84
n_embed = 384  #Number of embedding dimensions
n_layers = 8
n_heads = 8
dropout = 0.2
eval_iter = 20
max_iters = 10000
warmup_steps = 1000
max_lr = 3e-4
min_lr = 3e-5
resume_training = False
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_lr(it):
    # Linear warmup for warmup_iters steps
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

step = 0
checkpoints_dir = "checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)
if resume_training == True:
   checkpoint_dir = os.path.join(checkpoints_dir,"step_8000.pt")
   state_dict = torch.load(checkpoint_dir, map_location= device)
   model.load_state_dict(state_dict["model"])
   optimizer.load_state_dict(state_dict["optimizer_state_dict"])
   loss = state_dict["val_loss"]
   step = state_dict["step"]

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

while True:
    start_time = time.time()
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
    end_time = time.time()
    time_taken = end_time - start_time
    # Print step time and losses periodically
    if step % 100 == 0:
        validation_loss = get_validation_loss()
        print(f"step {step}, loss: {validation_loss:.4f}, lr: {lr:.6e}, time: {time_taken:.4f} seconds")
    # Saving checkpoint
    if step > 0 and step % 1000 == 0 :
      checkpoint_path = os.path.join(checkpoints_dir, f"step_{step}.pt")
      checkpoint = {
          'model': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'step': step,
          'val_loss': validation_loss
        }
      torch.save(checkpoint, checkpoint_path)
    if step == max_iters:
       break
    else:
       step += 1
