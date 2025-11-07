from typing import Tuple, List
import numpy as np
from dataclasses import dataclass

# ZenML imports
from zenml import pipeline, step
import os, math, time
from tinygrad import Tensor, nn, fetch, Device, TinyJit, GlobalCounters
import tiktoken


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    padded_vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention:
    def __init__(self, config:GPTConfig):
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.bias = Tensor.ones(1, 1, config.block_size, config.block_size).tril()
        self.bias.requires_grad = False

    def __call__(self, x:Tensor):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att.softmax()
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP:
    def __init__(self, config:GPTConfig):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x:Tensor) -> Tensor:
        return self.c_proj(self.c_fc(x).gelu())

class Block:
    def __init__(self, config:GPTConfig):
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def __call__(self, x:Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT:
    def __init__(self, config:GPTConfig):
        self.config = config

        self.wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def load_pretrained(self):
        weights = nn.state.torch_load(fetch(f'https://huggingface.co/gpt2/resolve/main/pytorch_model.bin'))
        transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
        for k in weights:
            if k == "wte.weight":
                weights[k] = weights[k].pad(((0, self.config.padded_vocab_size-self.config.vocab_size), (0,0))).to(None).contiguous()
            if k.endswith(transposed):
                weights[k] = weights[k].to(None).T.contiguous()
        # lm head and wte are tied
        weights['lm_head.weight'] = weights['wte.weight']
        nn.state.load_state_dict(self, weights)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            idx_next = logits.softmax().multinomial()
            idx = Tensor.cat(idx, idx_next, dim=1)
        return idx

    def __call__(self, idx:Tensor, targets=None):
        b, t = idx.shape
        pos = Tensor.arange(0, t, device=idx.device)

        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        x = self.ln_f(x.sequential(self.h))

        if targets is not None:
            logits = self.lm_head(x)[:, :, :self.config.vocab_size]
            loss = logits.sparse_categorical_crossentropy(targets)
        else:
            logits = self.lm_head(x[:, [-1], :])[:, :, :self.config.vocab_size]
            loss = None

        return logits, loss

@step
def start() -> List[int]:
    """Load the text dataset and return the tokens as a list."""
    # Load the tokens
    # prefer to use tiny_shakespeare if it's available, otherwise use tiny_stories
    # we're using val instead of train split just because it is smaller/faster
    tokens_bin = fetch("https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tiny_shakespeare_val.bin")
    assert os.path.isfile(tokens_bin)
    print(f"loading cached tokens in {tokens_bin}")
    with open(tokens_bin, "rb") as f:
        f.seek(0x400)
        tokens = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
    
    return tokens.tolist()

@step
def train(tokens: List[int], num_iterations: int, batch_size: int, sequence_length: int, gpus: int, skip_test: bool) -> Tuple[float, float]:
    """Finetune the GPT-2 model for a specified number of iterations."""
    B = batch_size
    T = sequence_length
    assert 1 <= T <= 1024
    
    # Initialize the model
    model = GPT(GPTConfig(n_layer=12, n_head=12, n_embd=768))
    model.load_pretrained()

    if gpus > 1:
        GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(gpus))
        for x in nn.state.get_parameters(model): x.to_(GPUS)  # we put a copy of the model on every GPU

    # init the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={""})
    decode = lambda l: enc.decode(l)

    # Convert tokens to Tensor
    tokens_tensor = Tensor(tokens)

    # lightweight dataloader
    def get_batch():
        assert B*T+1 <= len(tokens), "not enough tokens"
        # for 338,025 tokens. E.g. with B=8 T=1024, this will yield 41 batches before looping
        i = 0
        while True:
            x = tokens_tensor[i:i+B*T].view(B, T)
            y = tokens_tensor[i+1:i+B*T+1].view(B, T)
            yield x, y
            i += B*T
            if i + B*T + 1 >= len(tokens):
                i = 0 # in prod we'd want to randomize the start point a bit

    # forward backward for a few iterations
    data_iter = iter(get_batch())
    x, y = next(data_iter) # we'll overfit this batch below
    optimizer = nn.optim.AdamW(nn.state.get_parameters(model), lr=1e-4, weight_decay=0)

    print(f"model state:     {sum(x.nbytes() for x in nn.state.get_parameters(model))/1e9:.2f} GB")
    print(f"optimizer state: {sum(x.nbytes() for x in nn.state.get_parameters(optimizer))/1e9:.2f} GB")

    # shard the data on axis 0
    if gpus > 1: 
        x = x.shard(GPUS, axis=0)
        y = y.shard(GPUS, axis=0)

    @TinyJit
    @Tensor.train()
    def step(x:Tensor, y:Tensor) -> Tensor:
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        return loss.realize(*optimizer.schedule_step())

    for i in range(num_iterations):
        GlobalCounters.reset()
        t0 = time.perf_counter()
        loss = step(x.contiguous(), y.contiguous())
        Device[Device.DEFAULT].synchronize()
        t1 = time.perf_counter()
        print(f"iteration {i}, loss: {loss.item():.6f}, time: {(t1-t0)*1000:.3f}ms, {int(B*T/(t1-t0))} tok/s, {GlobalCounters.global_mem/1e9:.2f} GB")
    
    # For zenml, we'll return the final loss and model size as artifacts
    final_loss = loss.item()
    model_size = sum(x.nbytes() for x in nn.state.get_parameters(model))/1e9
    
    return final_loss, model_size

@step
def end(final_loss: float, model_size: float, gpus: int, batch_size: int, sequence_length: int, skip_test: bool) -> None:
    """Final step – report completion and optionally generate text."""
    print(f"Final loss: {final_loss:.6f}")
    print(f"Model size: {model_size:.2f} GB")

    # Generate text if not skipped
    if not skip_test:
        print("Generating text...")

        # Initialize the model for generation (similar to reference implementation)
        model = GPT(GPTConfig(n_layer=12, n_head=12, n_embd=768))
        model.load_pretrained()

        # Initialize tokenizer
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

        # Move model to default device if needed
        if gpus > 1:
            for x in nn.state.get_parameters(model):
                x.to_(Device.DEFAULT)

        # Generate text using the same approach as reference
        start = "<|endoftext|>"
        start_ids = encode(start)
        x = (Tensor(start_ids)[None, ...])
        max_new_tokens = 16
        temperature = 1.0

        y = model.generate(x, max_new_tokens, temperature=temperature)
        generated_text = decode(y[0].tolist())

        print(f"Generated text: '{generated_text}'")

    print("TinyGradLLMPipeline is all done - with text generation fix!")

@pipeline
def tinygrad_llm_pipeline(num_iterations: int = 10, batch_size: int = 4, sequence_length: int = 64, gpus: int = 1, skip_test: bool = False):
    tokens = start()
    final_loss, model_size = train(tokens=tokens, num_iterations=num_iterations, batch_size=batch_size, sequence_length=sequence_length, gpus=gpus, skip_test=skip_test)
    end(final_loss=final_loss, model_size=model_size, gpus=gpus, batch_size=batch_size, sequence_length=sequence_length, skip_test=skip_test)

if __name__ == "__main__":
    # Running the pipeline locally via ZenML's default orchestrator.
    tinygrad_llm_pipeline()
