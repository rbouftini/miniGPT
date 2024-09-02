from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

dataset = load_dataset("alielfilali01/Darija-Stories-Dataset", split="train")

text = ''.join([text for text in dataset["Text"]])

#Building the vocabulary
vocab = sorted(list(set(text)))
vocab_size = len(vocab)


#Tokenizing the text ~ Converting it to a sequence of integers according to our vocabulary
#Creating two dictionaries: one to represent a token into an unique integer
#Second to map back from the integer to the word
itoi = {index:val for index,val in enumerate(vocab)}
stoi = {val:index for index,val in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itoi[i] for i in l])

data = torch.tensor(encode(text), dtype= torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

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

def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data)- context_length, (batch_size,))
  x = torch.stack([data[i:i+context_length]for i in ix])
  y = torch.stack([data[i+1:i+1+context_length]for i in ix])
  x,y = x.to(device), y.to(device)
  return x,y

class Head(nn.Module):

  def __init__(self,head_dimension):
    super().__init__()
    self.query_head = nn.Linear(n_embed,head_dimension, bias=False)
    self.key_head = nn.Linear(n_embed, head_dimension, bias=False)
    self.vector_head = nn.Linear(n_embed, head_dimension, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
    self.dropout= nn.Dropout(dropout)

  def forward(self,x):
    B,T,C = x.shape
    q = self.query_head(x)
    k = self.key_head(x)
    v = self.vector_head(x)
    attention = q @ torch.transpose(k, -2, -1) * C**-0.5
    attention = torch.masked_fill(attention, self.tril[:T,:T]==0, float('-inf') )
    attention = F.softmax(attention, dim=-1)
    attention = self.dropout(attention)
    out = attention @ v
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, head_size, num_heads):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim =-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embed,4*n_embed),
        nn.ReLU(),
        nn.Linear(4*n_embed,n_embed),
        nn.Dropout(dropout)
    )
  def forward(self,x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embed, n_heads):
    super().__init__()
    head_size = n_embed // n_heads
    self.ma_head = MultiHeadAttention(head_size, n_heads)
    self.ffwd = FeedForward(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.ma_head(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embed) #Returns the token embeddings
    self.pos_embedding_table = nn.Embedding(context_length,n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_heads=n_heads)for _ in range(n_layers)])
    self.ln = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    #idx and targets are both (B,T) tensors
    B,T = idx.shape
    tok_embed = self.token_embedding_table(idx)                     #embedding object you give it a tensor and returns to you the embedding for each input #(B,T,n_embed)
    pos_embed = self.pos_embedding_table(torch.arange(T, device=device).expand(B,-1))
    #You did not have to do the expand because pytorch does broadcasting for unmatched dimentions
    x = tok_embed + pos_embed  #(B,T,C)
    x = self.blocks(x)
    x = self.ln(x)
    logits = self.lm_head(x)                #(B,T,vocab_size)

    if targets==None:
      return logits
    else:
      B,T,C = logits.shape                                         #logits are of shape B,T,C (where C is vocab_size representing the probabilities of each next token)
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  def generate(self, idx, max_new_tokens):
  # idx is (B,T) which is the current context we have in some batches
    for _ in range(max_new_tokens):
      # generating the predictions
      idx_cond = idx[:,-context_length:]
      logits = self(idx_cond)
      # get only the last token in the batch
      logits = logits[:,-1,:]
      probs = F.softmax(logits, dim=-1)                      #to sample from the distirbution we need a probability distirbution
      next_token = torch.multinomial(probs,num_samples=1)    #multiomial generates an index from the probability distribution
      idx = torch.cat((idx,next_token), dim = 1)
    return idx

model = BigramLanguageModel().to(device)

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
    out[split] = losses[iter].mean()
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

text = decode(model.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=10000)[0].tolist())

torch.save(model.state_dict(), 'checkpoint.pth')