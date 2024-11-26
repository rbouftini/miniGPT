from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# Muon optimizer   https://github.com/KellerJordan/Muon/blob/master/muon.py

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

#G should be bfloat16()
@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.float() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1): # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)

# -----------------------------------------------------------------------------


@dataclass
class MiniGPTConfig():
    vocab_size : int = 32209
    context_length : int = 512
    n_embed : int = 512  #Number of embedding dimensions
    n_layers : int = 8
    n_heads : int = 8

class RoPE(nn.Module):
  def __init__(self, head_size, base=10000):
    super().__init__()
    thetas = base ** (- 2 *torch.arange(0, (head_size)//2).float() / head_size)
    self.register_buffer("thetas", thetas)
    self.context_length_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, q):
    context_length = q.shape[1]
    dim = q.shape[3] // 2
    if self.context_length_cached != context_length:
      self.context_length_cached = context_length
      indexes = torch.arange(0,context_length, device= device).float()
      freqs = torch.outer(indexes,self.thetas).to(device)
      self.cos_cached = freqs.cos().view(1, context_length, 1, dim)
      self.sin_cached = freqs.sin().view(1, context_length, 1, dim)

  def rotate_embedding(self, v):
    d = v.shape[-1]//2
    v1 = v[..., :d]
    v2 = v[..., d:]
    d1 = v1 * self.cos_cached + v2 * self.sin_cached
    d2 = v1 * (-self.sin_cached) + v2 * self.cos_cached
    return torch.cat([d1, d2], dim=-1)

class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.n_heads = config.n_heads
    self.n_embed = config.n_embed
    self.head_size = self.n_embed // self.n_heads
    self.query_head = nn.Linear(self.n_embed, self.n_embed, bias= False)
    self.key_head = nn.Linear(self.n_embed, self.n_embed, bias=False)
    self.value_head = nn.Linear(self.n_embed, self.n_embed, bias= False)
    self.proj = nn.Linear(self.n_embed, self.n_embed, bias=False)
    self.rotary = RoPE(self.head_size)

  def forward(self, x):
    B,T,E = x.shape
    q = self.query_head(x).view(B, T, self.n_heads, self.head_size)
    k = self.key_head(x).view(B, T, self.n_heads, self.head_size)
    v = self.value_head(x).view(B, T, self.n_heads, self.head_size)
    self.rotary(q)
    q = self.rotary.rotate_embedding(q)
    v = self.rotary.rotate_embedding(v)
    out = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True)
    out = out.transpose(1, 2).contiguous().view(B,T,E)
    out = self.proj(out)
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embed):
    super().__init__()
    self.l1 = nn.Linear(n_embed,4*n_embed, bias=False)
    self.l2 = nn.Linear(4*n_embed,n_embed, bias=False)

  def forward(self,x):
    x = self.l1(x)
    x = F.relu(x).square()
    x = self.l2(x)
    return x

class Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.ma_head = MultiHeadAttention(config)
    self.ffwd = FeedForward(config.n_embed)
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
    x = self.token_embedding_table(idx)                     #embedding object you give it a tensor and returns to you the embedding for each input #(B,T,n_embed)
    x = self.blocks(x)
    x = self.ln(x)
    logits = self.lm_head(x)                #(B,T,vocab_size)
    logits = 30 * torch.tanh(logits / 30)
    logits = logits.float()

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

  def configure_optimizer(self, weight_decay, learning_rate_adam, learning_rate_muon, betas, momentum_muon):
    param_dict_adam = {}
    param_dict_muon = {}
    for name, param in self.named_parameters():
      if ("blocks" not in name) or param.dim() < 2 :
        param_dict_adam[name] = param
      else:
        param_dict_muon[name] = param  

    decay_params_adam = [param for name, param in param_dict_adam.items() if param.dim() >= 2 ]
    nodecay_params_adam = [param for name, param in param_dict_adam.items() if param.dim() < 2]
    decay_params_muon = [param for name, param in param_dict_muon.items()]   

    optim_groups_adam = [
        {'params': decay_params_adam, 'weight_decay': weight_decay},
        {'params': nodecay_params_adam, 'weight_decay': 0.0}
    ]

    optim_groups_muon = [
        {'params': decay_params_muon, 'weight_decay': weight_decay}
    ]

    optimizer1 = torch.optim.AdamW(optim_groups_adam,lr=learning_rate_adam, betas=betas, eps=1e-8, fused= True)
    optimizer2 = Muon(optim_groups_muon, lr=learning_rate_muon, momentum= momentum_muon)
    optimizers = [optimizer1, optimizer2]

    return optimizers
