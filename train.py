import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
from model import MiniGPTConfig, MiniGPT

torch.manual_seed(1337)

context_length = 512
batch_size = 48
n_embed = 384  #Number of embedding dimensions
n_layers = 8
n_heads = 8
dropout = 0.2
eval_iter = 20
max_iters = 20000
warmup_steps = 1000
max_lr = 6e-4
min_lr = 6e-5
vocab_size = 16384
weight_decay = 0.1
resume_training = False
device = "cuda" if torch.cuda.is_available() else "cpu"

def configure_optimizer(self, weight_decay, learning_rate, betas):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups,learning_rate=learning_rate, betas=betas, eps=1e-8, fused= True)

        return optimizer

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

model_parameters = dict( vocab_size = vocab_size, context_length = context_length, n_embed = n_embed, n_layers = n_layers,
    n_heads = n_heads, dropout = dropout)

def get_batch(split):
  if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
  else:
        data = np.memmap('val.bin', dtype=np.uint16, mode='r')
  ix = torch.randint(len(data)- context_length, (batch_size,))
  x = torch.stack([torch.from_numpy((data[i:i+context_length]).astype(np.int64)) for i in ix])
  y = torch.stack([torch.from_numpy((data[i+1:i+1+context_length]).astype(np.int64)) for i in ix])
  if device == 'cuda':
        # pin arrays x,y to be loaded in main memory, and move then to GPU asynchronously
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
  else:
        x, y = x.to(device), y.to(device)
  return x, y

config = MiniGPTConfig(**model_parameters)
model = MiniGPT(config).to(device)
model = torch.compile(model)

optimizer = configure_optimizer(weight_decay, learning_rate=max_lr, betas=(0.9,0.95))
scaler = torch.amp.GradScaler(device=device)

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

train_loss = 0
while True:
    start_time = time.time()
    xb, yb = get_batch("train")
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(xb, yb)
    train_loss += loss
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
        print(f"step {step}, train loss:{train_loss/100:.4f},  validation loss: {validation_loss:.4f}, lr: {lr:.6e}, time: {time_taken:.4f} seconds")
        train_loss = 0
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
