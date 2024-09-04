from datasets import load_dataset
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MiniGPTConfig, MiniGPT

torch.manual_seed(1337)

context_length = 512
batch_size = 64
n_embed = 384  #Number of embedding dimensions
n_layers = 6
n_heads = 6
dropout = 0.2
eval_iter = 200
max_iters = 5001
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset("alielfilali01/Darija-Stories-Dataset", split="train")

text = ''.join([text for text in dataset["Text"]])

#Building the vocabulary
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

model_parameters = dict( vocab_size = vocab_size, context_length = context_length, n_embed = n_embed, n_layers = n_layers,
    n_heads = n_heads, dropout = dropout)

#Tokenizing the text ~ Converting it to a sequence of integers according to our vocabulary
#Creating two dictionaries: one to represent a token into an unique integer
#Second to map back from the integer to the word
itoi = {index:val for index,val in enumerate(vocab)}
stoi = {val:index for index,val in enumerate(vocab)}

tokenizers = {"itoi": itoi, "stoi" :stoi}
with open("tokenizers.pkl", "w") as f:
  pickle.dump(tokenizers, f)
  
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itoi[i] for i in l])

data = torch.tensor(encode(text), dtype= torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data)- context_length, (batch_size,))
  x = torch.stack([data[i:i+context_length]for i in ix])
  y = torch.stack([data[i+1:i+1+context_length]for i in ix])
  x,y = x.to(device), y.to(device)
  return x,y

config = MiniGPTConfig(**model_parameters)
model = MiniGPT(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) 

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ["train", "val"]:
    losses = torch.zeros(eval_iter)
    for iter in range(eval_iter):
      xb, yb = get_batch(split)
      logits, loss = model(xb,yb)
      losses[iter] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

for steps in range(max_iters):
  xb, yb = get_batch("train")
  if steps % eval_iter ==0:
    losses = estimate_loss()
    print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  logits, loss = model(xb,yb)
  # Zerowing out gradients from previous step
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

#text = decode(model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=10000)[0].tolist())

torch.save(model.state_dict(), 'checkpoint.pth')

