from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel
import tiktoken
import inspect
import time

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <endoftext>
    n_layer: int = 12 # number of transformer blocks
    n_head: int = 12 # number of heads
    n_embd: int = 768 #embedding dimension



class DataLoaderLite:
    def __init__(self, B, T, split='train', train_frac=0.9):
        self.B = B
        self.T = T
        assert split in {"train", "val"}

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        tokens = torch.tensor(tokens)
        n = int(train_frac * len(tokens))
        if split == 'train':
            self.tokens = tokens[:n]
        else:
            self.tokens = tokens[n:]
        print(f"loaded {len(self.tokens)} tokens for split '{split}'")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    # missing randomness for next batch so trainning more than 1 epoch can be problematic
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a bias, but a mask to ensure that attention is only applied to the left in the input sequence
        # dont need this with flash attention, it has a built-in causal mask
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh = number of heads, hs = head size and C (number of channels) = n_embd = nh * hs
        qkv = self.c_attn(x) # (B,T,3C)
        q,k,v = qkv.split(self.n_embd,dim=2) #split into (B,T,C) = (B,T,nh*hs) - concatenate all attention heads
        k = k.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        q = q.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        v = v.view(B,T,self.n_head,C // self.n_head).transpose(1,2) # (B,nh,T,hs)
        # atteention matrix for all queries and keys
        # att = (q @ k.transpose(-2,-1)) *(1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att,dim=-1)
        # y = att @ v #(B,nh,T,T) x * (B,nh,T,hs) -> (B,nh,T,hs)
        # we can apply Flash attention for faster computation and lower memory usage
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C) #.contiguous() makes sure the tensor’s memory layout is compact and consistent after a transpose (or similar ops). It’s basically a "safety step" before .view().
        #output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # smoother RELU - only use approximation for faster computation
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) #layer normalization before self-attention in contrast to after in the self-attention paper
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x)) #residual connection
        x = x + self.mlp(self.ln_2(x)) #residual connection
        return x

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #weights of token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), #weights of positional embeddings
            drop = nn.Dropout(0.1),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd), #final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight #copy reference

        #initialize weights
        self.apply(self.__init_weights)
    
    def __init_weights(self,module):
        #0.02 is coming from 1/sqrt(n_embd) = 1/sqrt(1600) in the gpt2 model paper
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # standard deviation grows inside the residual connection stream
                # hence, we scaledown the initialization of the final projection to compensate
                std = std * (2*self.config.n_layer) ** -0.5
                # we use 2 times since we apply 2 residual connections per block layer - see forward of Block class
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) #0 bias
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        hf_keys = set(sd_keys_hf)
        our_keys = set(sd_keys)

        missing_in_hf = our_keys - hf_keys
        missing_in_our = hf_keys - our_keys

        print("Missing in HF:", missing_in_hf)
        print("Missing in our GPT:", missing_in_our)

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no. - this is like L2 regularization that penalizes large weights
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

if __name__ == "__main__":
    device = 'cuda'
    num_return_sequences = 5
    max_length = 30
    max_lr = 3e-4
    min_lr = max_lr *0.1
    def get_lr(it,warmup_steps = 10,max_steps = 50):
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0.0 <= decay_ratio <= 1.0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


    model = GPT(GPTConfig())
    model.train()
    model.to(device)
    # only work in ubuntu linux
    #model = torch.compile(model) # makes your model run faster by compiling it, instead of interpreting it op-by-op in Python

    train_loader = DataLoaderLite(B=4,T=32)
    val_loader = DataLoaderLite(B=4,T=32,split='val')
    torch.set_float32_matmul_precision('high') #default is highest
    # optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas = (0.9,0.95),eps=1e-8) #follow GPT3 hyperparameters
    optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=max_lr,device_type=device)
    for step in range(1000):
        t0 = time.time()
        x,y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits,loss = model(x,y)
        # default dtype is float32 - may not be too efficient -switch to float16
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0) #clip the gradient to 1.0
        # prevent the gradient from exploding
        # determine annd set the learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"step{step}:loss:{loss.item()}. time per iter: {(t1 - t0)*1000:.2f}ms,tok/sec:{(x.shape[0]*x.shape[1])/(t1 - t0):.2f},grad_norm:{norm:.4f} ")
