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
batch_size = 32
n_embed = 512  #Number of embedding dimensions
n_layers = 8
n_heads = 8
eval_steps = 20
max_steps = 2000 * 4 # Training for 4 epochs
warmup_steps = 0
warmdown_steps = 400
learning_rate_adam = 3e-3
learning_rate_muon = 3e-4
vocab_size = 32209
weight_decay = 0.1
resume_training = False
total_batch_size = 100000
grad_accum_steps = total_batch_size // (context_length * batch_size)
best_val_loss = float("+inf")
device = "cuda" if torch.cuda.is_available() else "cpu"

# learning rate decay scheduler WSD
def get_lr(it, lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return lr * (it+1) / warmup_steps
    # 2) Stable learning rate
    if it < max_steps - warmdown_steps:
        return lr
    # 3) Decay learning rate
    else:
      decay_ratio = (max_steps - it) / warmdown_steps
      return lr * decay_ratio

model_parameters = dict( vocab_size = vocab_size, context_length = context_length, n_embed = n_embed, n_layers = n_layers,
    n_heads = n_heads)

class DataLoader:
    def __init__(self, file_name, batch_size, context_length):
      self.data = np.memmap(file_name, dtype=np.uint16, mode='r+')
      self.n_examples = len(self.data)
      self.current = 0
      self.batch_size = batch_size
      self.context_length = context_length
      self.full_context = self.batch_size * self.context_length
      self.n_valid_sequences = (self.n_examples - 1) // self.full_context
      self.indexes = np.arange(0, self.n_valid_sequences * self.full_context, self.full_context)

    def shuffle(self):
      np.random.shuffle(self.indexes)

    def get_batch(self):
      if self.current == self.n_valid_sequences:
        self.current = 0
        self.shuffle()

      idx = self.indexes[self.current]
      x = torch.from_numpy((self.data[idx: idx + self.full_context]).astype(np.int64)).view(-1,self.context_length)
      y = torch.from_numpy((self.data[idx+1 : idx + self.full_context+1]).astype(np.int64)).view(-1,self.context_length)
      self.current += 1

      x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

      return x,y

config = MiniGPTConfig(**model_parameters)
model = MiniGPT(config).to(device)
train_loader = DataLoader("train.bin", batch_size, context_length)
val_loader = DataLoader("val.bin", batch_size, context_length)

optimizers = model.configure_optimizer(weight_decay, learning_rate_adam=learning_rate_adam, learning_rate_muon= learning_rate_muon, betas=(0.9,0.95), momentum_muon=0.95)

model = torch.compile(model)
scaler = torch.amp.GradScaler(device=device)

step = 0
checkpoints_dir = "checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)

if resume_training == True:
   checkpoint_dir = os.path.join(checkpoints_dir,"step_500.pt")
   state_dict = torch.load(checkpoint_dir, map_location= device)
   model.load_state_dict(state_dict["model"])
   for optimizer, state in zip(optimizers, state_dict["optimizer_states"]):
        optimizer.load_state_dict(state)
   loss = state_dict["val_loss"]
   step = state_dict["step"]

@torch.no_grad()
def get_validation_loss():
  model.eval()
  total_loss = 0.0
  for iter in range(eval_steps):
    xb, yb = val_loader.get_batch()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      _ , loss = model(xb,yb)
    total_loss += loss.item()
  total_loss /= eval_steps
  model.train()
  return total_loss

while True:
    train_loss = 0.0
    start_time = time.time()
    
    for small_grad_step in range(grad_accum_steps):
        xb, yb = train_loader.get_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(xb, yb)
        loss = loss / grad_accum_steps
        train_loss += loss.detach()
        loss.backward()

    for i, optimizer in enumerate(optimizers):
      lr = learning_rate_adam if i==0 else learning_rate_muon
      for param_group in optimizer.param_groups:
        param_group["lr"] = get_lr(step, lr)
      
    #Gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    for optimizer in optimizers:
      optimizer.step()

    # Zeroing out gradients from the previous step
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    end_time = time.time()
    time_taken = end_time - start_time

    # Print step time and losses periodically
    if step % 20 == 0:
      validation_loss = get_validation_loss()
      print(f"step {step}, train loss:{train_loss:.4f},  validation loss: {validation_loss:.4f}, time: {time_taken:.4f} seconds")

    # Saving checkpoint
    if validation_loss < best_val_loss:
      best_val_loss = validation_loss
      print("Saving checkpoint...")
      checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint.pt")
      checkpoint = {
          'model': model.state_dict(),
          'optimizer_states': [opt.state_dict() for opt in optimizers],
          'step': step,
          'val_loss': validation_loss
        }
      torch.save(checkpoint, checkpoint_path)

    if step == max_steps:
       break
    else:
       step += 1
