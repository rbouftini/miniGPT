from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class MiniGPTConfig():
    vocab_size : int = 16384
    context_length : int = 512
    n_embed : int = 384  #Number of embedding dimensions
    n_layers : int = 8
    n_heads : int = 8
    dropout : float = 0.2

    
class Head(nn.Module):

  def __init__(self,head_dimension, config):
    super().__init__()
    self.query_head = nn.Linear(config.n_embed,head_dimension, bias=False)
    self.key_head = nn.Linear(config.n_embed, head_dimension, bias=False)
    self.vector_head = nn.Linear(config.n_embed, head_dimension, bias=False)
    self.dropout= nn.Dropout(config.dropout)

  def forward(self,x):
    B,T,C = x.shape
    q = self.query_head(x)
    k = self.key_head(x)
    v = self.vector_head(x)
    #Flash Attention
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out  = self.dropout(out)
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self,head_size, config):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size, config) for _ in range(config.n_heads)])
    self.proj = nn.Linear(config.n_embed, config.n_embed)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim =-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embed, dropout):
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
  def __init__(self,config):
    super().__init__()
    head_size = config.n_embed // config.n_heads
    self.ma_head = MultiHeadAttention(head_size, config)
    self.ffwd = FeedForward(config.n_embed, config.dropout)
    self.ln1 = nn.LayerNorm(config.n_embed)
    self.ln2 = nn.LayerNorm(config.n_embed)

  def forward(self, x):
    x = x + self.ma_head(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class MiniGPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.token_embedding_table = nn.Embedding(config.vocab_size,config.n_embed) #Returns the token embeddings
    self.pos_embedding_table = nn.Embedding(config.context_length,config.n_embed)
    self.blocks = nn.Sequential(*[Block(config)for _ in range(config.n_layers)])
    self.ln = nn.LayerNorm(config.n_embed)
    self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
    self.config = config
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      module.weight.data.normal_(mean=0.0, std=0.05)
      if module.bias is not None:
        module.bias.data.zero_()

    elif isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0.0, std=0.05)


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
      idx_cond = idx[:,-self.config.context_length:]
      logits = self(idx_cond)
      # get only the last token in the batch
      logits = logits[:,-1,:]
      probs = F.softmax(logits, dim=-1)                      #to sample from the distirbution we need a probability distirbution
      next_token = torch.multinomial(probs,num_samples=1)    #multiomial generates an index from the probability distribution
      idx = torch.cat((idx,next_token), dim = 1)
    return idx
