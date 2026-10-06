# import packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# initialize variables
context_size = 8
batch_size = 4
eval_iter = 200
max_iter = 3000
eval_interval = 300
lr = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the data
text = open("input.txt").read()
n90 = int(len(text) * 0.90)
train_set = text[:n90]
val_set = text[n90:]

# make tokens
itos = {token: ch for token, ch in enumerate(sorted(list(set(text))))}
stoi = {ch: token for token, ch in itos.items()}
vocab_size = len(itos)
encode = lambda x: torch.tensor([stoi[ch] for ch in x])
decode = lambda x: "".join([itos[token.item()] for token in x])

# Helper function
def get_data(type="train"):
    data = train_set if type == "train" else val_set
    idxs = torch.randint(len(data)-context_size-1, (batch_size, ))
    x = torch.stack([encode(data[idx:idx+context_size]) for idx in idxs])
    y = torch.stack([encode(data[idx+1:idx+context_size+1]) for idx in idxs])

    return x.to(device), y.to(device)

# make Biagram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        # idx - B, T    target - B, T
        logits = self.token_embeddings(idx) # B, T, C
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

    def generate(self, idx, max_new_tokens):
        # idx - B, T            max_new_tokesn - int
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # logits - B, T, C but we will generate new character on based of last one
            # so we will see last token of logits only
            logits = logits[:, -1, :] # B, C
            probs = F.softmax(logits, dim=1) # B, C
            new_idx = torch.multinomial(probs, num_samples=1) # 1, 
            idx = torch.cat([idx, new_idx], dim=1) # B, T+1

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
        print(f"Step {i}: Train Loss {losses["train"]:.4f} Val Loss {losses["val"]:.4f}")

    # forward pass    
    x, y = get_data("train")
    _, loss = m(x, y)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # update parameters
    optimizer.step()

print(decode(m.generate(torch.ones(1,1, dtype=torch.long, device=device), 500)[0]))