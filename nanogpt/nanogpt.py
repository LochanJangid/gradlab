# import packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters ----------------
context_size = 256 # Before 8
batch_size = 64 # Before 4
eval_iter = 200# Before 200
max_iter = 5000 # Before 3000
eval_interval = 500# Before 300
lr = 3e-4 # Before 1e-3
n_emb = 384 # Before 32
device = "cuda" if torch.cuda.is_available() else "cpu"
head_size = 6 # Before 4
n_layer = 6 # Before 3
dropout = 0.2 # nn.Dropout(dropout) think dropout as regularization technique where he shut off many neurons and make model less noisy
# --------------------------------

# Get the data
text = open("input.txt", "r", encoding="utf-8").read()


# make tokens
itos = {token: ch for token, ch in enumerate(sorted(list(set(text))))}
stoi = {ch: token for token, ch in itos.items()}
vocab_size = len(itos)
encode = lambda x: torch.tensor([stoi[ch] for ch in x])
decode = lambda x: "".join([itos[token.item()] for token in x])

# Encode everything once safe computation
data_all = encode(text)
n90 = int(len(text) * 0.90)
train_set = data_all[:n90]
val_set = data_all[n90:]

# Helper function
def get_data(type="train"):
    data = train_set if type == "train" else val_set
    idxs = torch.randint(len(data)-context_size-1, (batch_size, ))
    offsets = torch.arange(context_size)
    x = data[idxs.unsqueeze(dim=1) + offsets]
    y = data[idxs.unsqueeze(dim=1) + offsets + 1]
    
    return x.to(device), y.to(device)

# make a single head attention
class Head(nn.Module):
    def __init__(self, n_emb, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        # self.tril = torch.tril(torch.ones(context_size, context_size, device=device))
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size))) #  it's good to save in buffer so it will add into state_dict and no need to explicitly say device=device
        self.dropout = nn.Dropout(dropout)
    # Attention to shapes :p
    # tril - context_size, context_size
    # x - B, T, C=n_emb
    # Q - B, T, head_size
    # k - B, T, head_size
    # v - B, T, head_size
    # wei - (B, T, head_size) @ (B, head_size, T) -> B, T, T
    # xbow - (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
    
    def forward(self, x):
        B, T, C = x.shape
        Q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wei = Q @ k.transpose(-2, -1) * self.head_size ** -0.5
        
        wei = torch.masked_fill(wei, self.tril[:T, :T] == 0, float("-inf")) # [:T, :T] is for slice to the current length and be flexible to T <= context_length
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        xbow = wei @ v
        return xbow

# Now we will go with multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_emb, head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(out)
        x = self.dropout(x)
        return x


# Head + MultiHeadAttention --> Run multiple heads in parallel so the computation will become fast and great use of GPU | i don't have one just using kaggle notebooks:p
class ParallelHeadAttention(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        # RULE: C = head_size * n_head (Everytime don't matter how they run) in our transformer C is n_emb
        self.nh = n_head # number of heads
        self.hs = n_emb//n_head # head size

        self.query = nn.Linear(n_emb, self.nh*self.hs, bias=False) # self.nh*self = n_emb (just for understanding what we are doing...)
        self.key = nn.Linear(n_emb, self.nh*self.hs, bias=False)
        self.value = nn.Linear(n_emb, self.nh*self.hs, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        self.proj = nn.Linear(n_emb, n_emb, bias=False)
        self.dropout = nn.Dropout(dropout)

    # ! Please give some attention to shapes  hahahahahahahahahaha Very Complicated but that's fun :p
    # x - B, T, C
    # Q - Shapes step by step
    #    B, T, C # first shape
    #    B, T, nh, hs # split C into nh, hs 
    #    B, nh, T, hs # make nh higher dimension because of matrix multiplication (in pytorch matrix multiplication consider last 2 dimensions)
    # k - 
    #    B, T, C -> B, T, nh, hs -> B, nh, T, hs # just same as Q
    #    B, nh, hs, T # transposed shape of k for `Q @ k.T`  [use k.transpose(-1, -2)]
    # v - 
    #    B, T, C -> B, T, nh, hs -> B, nh, T, hs # just same as Q but it is good for (wei @ v) becauzeeeee.... see out shapes :)
    # wei = Q * k.T - (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
    # out = wei @ v - (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    # Reassemble the head back so change shape of out to get back out main shape to stay consistant in shape
    # out - (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, nh*hs) that is also identical to (B, T, C)
    
    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x) # B, T, C
        Q = Q.view(B, T, self.nh, self.hs) # B, T, nh, hs
        Q = Q.transpose(1, 2) # B, nh, T, hs

        k = self.key(x).view(B, T, self.nh, self.hs) 
        k = k.transpose(1, 2)

        v = self.value(x).view(B, T, self.nh, self.hs)
        v = v.transpose(1, 2)

        wei = Q @ k.transpose(-1, -2) * self.hs ** -0.5 # B, nh, T, T
        # Masking and averaging (if you know you know)
        wei = torch.masked_fill(wei, self.tril[:T, :T] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei) # faaaaaaaaaaaaaaa

        out = wei @ v # B, nh, T, hs
        out = out.transpose(1, 2) # B, T, nh, hs
        out = out.contiguous().view(B, T, self.nh*self.hs) # B, T, nh*hs # used contiguous() becauze after transpose we can't do view directly becauze transpose return non-contiguous tensor 

        out = self.proj(out)        
        out = self.dropout(out)
        return out      

# Feed-Forward just MLP - it gives the space or time to think about collected information
# now tokens think individually on learnt information
class FeedForward(nn.Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        hidden = 4*fan_out
        self.model = nn.Sequential(
            nn.Linear(fan_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, fan_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.model(x)

# Make a block of transformer where we have communication with computation
# means communication (multi head attention) + computation (feed forward)
class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        # self.head = MultiHeadAttention(head_size, n_emb//head_size)
        self.head = ParallelHeadAttention(n_emb, n_head)
        self.ffd = FeedForward(n_emb, n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        # make residual connection by fork off 
        # here we do layer normalization before attention head & feed forward 
        # it is slighlty different from original transformer architecture but it is better :)
        x = x + self.head(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x

# make Biagram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_emb)
        self.position_embeddings = nn.Embedding(context_size, n_emb)
        # self.sa_head = Head(n_emb, n_emb) # single attention head
        
        # the both line replicated by `Block`
        # self.ma_head = MultiHeadAttention(4, n_emb//4) # Multi Attention Head
        # self.ffd = FeedForward(n_emb, n_emb) # Feed forward (Linear + Relu)

        self.blocks = nn.Sequential(*[Block(n_emb, n_head) for _ in range(n_layer)])
        self.ln_f =    nn.LayerNorm(n_emb) # final layernorm: a another layer normalization before going outside of transformer block
        self.lm_head = nn.Linear(n_emb, vocab_size)
        

    def forward(self, idx, target=None):
        # idx - B, T    target - B, T
        B, T = idx.shape
        token_emb = self.token_embeddings(idx) # B, T, C=n_emb
        pos_emb = self.position_embeddings(torch.arange(T, device=device)) # T, C=n_emb
        x = token_emb + pos_emb # (B, T, C) + (T, C) [broadcast to (B, T, C)] --. B, T, C=n_emb
        # xbow = self.sa_head(x) # (B, T, C=n_emb) --> (B, T, head_size) [: for now head_size is identical to n_emb ]
        
        # these two lines replicated by `Block` sorry blocks :p
        # xbow = self.ma_head(x)
        # xact = self.ffd(xbow)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # B, T, C=n_emb -----Linear Layer----->  B, T, C=vocab_size 


        if target is None:
            loss = None
        else:
            # Cross entropy take B, C but we have B, T, C so we need to stretch out B, T 
            # into B*T sequqnetially so it become B*T, C
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target) # 1,

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx - B, T            max_new_tokesn - int
        self.eval()
        for _ in range(max_new_tokens):
            # crop context to max context size
            idx_cond = idx[:, -context_size:]
            logits, _ = self(idx_cond)
            # logits - B, T, C but we will generate new character on based of last one
            # so we will see last token of logits only
            logits = logits[:, -1, :] # B, C
            probs = F.softmax(logits, dim=1) # B, C
            new_idx = torch.multinomial(probs, num_samples=1) # 1, 
            idx = torch.cat([idx, new_idx], dim=1) # B, T+1
        self.train()
        return idx

# Function for finding estimate loss on another more samples so we will not distracted by noisy batches false losses
@torch.no_grad()
def estimate_loss(m):
    res = {}
    m.eval() # Set the module in evaluation mode    
    for type in ["train", "val"]:
        out = torch.zeros(eval_iter)
        for i in range(eval_iter):
            x, y = get_data(type)
            logits, loss = m(x, y)
            out[i] = loss.item()
        res[type] = out.mean().item()
    m.train() # reset module to train mode
    return res

# instantiate bigram model
m = BigramLanguageModel(vocab_size)
m = m.to(device)

# create Adam optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

# Train bigram model
for i in range(max_iter):

    # get metric about model once in a while
    if i % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"Step {i}: Train Loss {losses['train']:.4f} Val Loss {losses['val']:.4f}")

    # forward pass    
    x, y = get_data("train")
    _, loss = m(x, y)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # update parameters
    optimizer.step()

print(decode(m.generate(torch.ones(1,1, dtype=torch.long, device=device), 500)[0]))