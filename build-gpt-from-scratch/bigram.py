'''
Example usage:

python build-gpt-from-scratch/bigram.py
'''

import torch
import torch.nn as nn
from torch.nn import functional as F

#################
# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
#################

torch.manual_seed(1337)

## Pre-process data
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('build-gpt-from-scratch/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
encode = lambda sentence: [stoi[c] for c in sentence]
decode = lambda tokens: ''.join([itos[i] for i in tokens])
data = torch.tensor(encode(text), dtype=torch.long)
# Train / validation data split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generates a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # ix is the starting index of the context window; generate `batch_size` such
    # indexes in parallel, 1 per batch.
    # ix is a 1-d tensor of `block_size` elements in 1st dimension.
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Let's say x == [1, 2, 3], then in a single batch, we have 3 samples (aka 3
    # context windows), and for each, the expected target is the next element:
    # [1] -> 2
    # [1, 2] -> 3
    # [1, 2, 3] -> 4
    #
    # From x's perspective, [data[i:i+block_size] for i in ix] is a 2d tensor,
    # but it really contains 3-dimensional information:
    # * sampling start index, randomized via torch.randint (1st tensor dim)
    # * context window length, varies from 1 to block_size (2nd tensor dim)
    # * sub-window per context length (not explicitly a tensor dim)
    #
    # IOW, each element in the 2d x tensor is an *array* representing
    # *full* context window. This single element can be expanded to exactly
    # block_size sub context windows aka sub arrays.
    #
    # From y's perspective, it's also a 2d tensor, but each element is a scalar
    # not an array. i-th scalar in y is the target for a sub context window.
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

# This context manager hints pytorch that the body doesn't participate in
# gradient descent, so that pytorch knows not to store intermediate gradients
# which can be efficient at computation.
@torch.no_grad()
def estimate_loss():
    # purely running inference to evaluate loss, without training the model
    out = {}
    model.eval()  # enter eval mode, aka inference
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # enter training mode
    return out

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is not a parameter of the pytorch module (i.e. should not be part
        # of back propagation); pytorch convention is to stage under a buffer.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,C)@(B,C,T)->(B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B,T,T)@(B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """Multile heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])
        # for residule propagation, at the end of multi-head attention, we need
        # a projection to sum the residule and the multi-head attention output.
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # Concat over channels dimension, which is last dimension.
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads * head_size)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # times 4 to match the MLP in original attention paper.
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # projection layer for residual propagation.
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attn) then computation (ffwd)."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        # x + ... to implement residual block propagation.
        x = x + self.sa(x)
        x = x + self.ffwd(x)
        return x


class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a
        # lookup table.
        #
        # Embidding is vocab_size by n_embd, not vocab_size by vocab_size,
        # we want a layer of indirection, to get *token* embeddings, instead of
        # *logit* embeddings.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # positional encoding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # # self-attention
        # self.sa_heads = MultiHeadAttention(4, n_embd//4)  # 4 heads of 8-dimensional self-attention
        # # final MLP layer
        # self.ffwd = FeedForward(n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )
        # to go from token embeddings to logits, we need another linear layer.
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of ints.
        tok_emb = self.token_embedding_table(idx)  # (B, T, C), C=num embeddings=n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # leverage broadcast
        # x = self.sa_heads(x)  # apply only one head attention.
        # x = self.ffwd(x)  # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B, T, C), C=vocab_size
        
        # logits is one round of inference. we can then evaluate loss.
        # small quirk: pytorch wants channel dimension to come as 2nd dimension,
        # i.e. (B*T, C), not (B, T, C).
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            return logits, None
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indicies of the current context.
        for _ in range(max_new_tokens):
            # get the predictions
            # 
            # ATTN: idx[:, -block_size:], because for positional embedding, we
            # only have a lookup table up to block_size, so for inference we
            # must be consistent to have at most block_size tokens in the
            # inference context window.
            logits, _ = self(idx[:, -block_size:])
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# Now train the model.
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for iter in range(max_iters):
    # every once in a while, collect and report training loss.
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f'step {iter}: train loss {losses["train"]}, \
                val loss {losses["val"]}')
            
    # training
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
def demo():
    print(
        decode(
            m.generate(
                torch.zeros((1, 1), dtype=torch.long), 
                max_new_tokens=500)[0].tolist()))
demo()