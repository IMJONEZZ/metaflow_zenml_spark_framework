from metaflow.flowspec import FlowSpec
from metaflow.decorators import step
from metaflow.parameters import Parameter
from dataclasses import dataclass
import os, math, time

try:
    import numpy as np
    from tinygrad import Tensor, nn, fetch, Device, TinyJit, GlobalCounters
    import tiktoken
except ImportError as e:
    print(f"Import error: {e}")
    
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
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.bias = Tensor.ones(1, 1, config.block_size, config.block_size).tril()
        self.bias.requires_grad = False

    def __call__(self, x:Tensor):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = att.softmax()
        y = att @ v
        y = y.transpose(1, 2).view(B, T, C)
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
        self.wte.weight = self.lm_head.weight

    def load_pretrained(self):
        weights = nn.state.torch_load(fetch('https://huggingface.co/gpt2/resolve/main/pytorch_model.bin'))
        transposed = ('attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight')
        for k in weights:
            if k == "wte.weight":
                weights[k] = weights[k].pad(((0, self.config.padded_vocab_size-self.config.vocab_size), (0,0))).to(None).contiguous()
            if k.endswith(transposed):
                weights[k] = weights[k].to(None).T.contiguous()
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
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        x = self.ln_f(x.sequential(self.h))

        if targets is not None:
            logits = self.lm_head(x)[:, :, :self.config.vocab_size]
            loss = logits.sparse_categorical_crossentropy(targets)
        else:
            logits = self.lm_head(x[:, [-1], :])[:, :, :self.config.vocab_size]
            loss = None

        return logits, loss

class TinyGradLLMFlow(FlowSpec):
    """A Metaflow flow that finetunes a GPT-2 model on a text dataset using tinygrad.

    This flow consists of the following steps:
        1. start - load and preprocess the text dataset.
        2. train - finetune the GPT-2 model for a specified number of iterations.
        3. end - final step indicating completion with optional text generation.
    """

    # Number of training iterations (default 10)
    num_iterations = Parameter("num_iterations", default=10, type=int)  # type: ignore
    # Batch size (default 4)
    batch_size = Parameter("batch_size", default=4, type=int)  # type: ignore
    # Sequence length (default 64)
    sequence_length = Parameter("sequence_length", default=64, type=int)  # type: ignore
    # Skip test generation (default False)
    skip_test = Parameter("skip_test", default=False)
    # Number of GPUs to use (default 1 for GPU training, but text generation handles fallbacks)
    gpus = Parameter("gpus", default=1, type=int)  # type: ignore

    @step
    def start(self):
        """Load the text dataset and return the training dataloader."""
        self.B = int(self.batch_size)  # type: ignore
        self.T = int(self.sequence_length)  # type: ignore
        T_val = int(self.T)
        assert 1 <= T_val <= 1024
        
        tokens_bin = fetch("https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tiny_shakespeare_val.bin")
        assert os.path.isfile(tokens_bin)
        print(f"loading cached tokens in {tokens_bin}")
        with open(tokens_bin, "rb") as f:
            f.seek(0x400)
            tokens = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
        
        self.tokens = Tensor(tokens)
        
        enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        self.decode = lambda l: enc.decode(l)
        
        self.next(self.train)
    
    @step
    def train(self):
        """Finetune the GPT-2 model for a specified number of iterations."""
        B = int(self.B)  # type: ignore
        T = int(self.T)  # type: ignore
        gpus = int(self.gpus)  # type: ignore
        num_iterations = int(self.num_iterations)  # type: ignore
        
        self.model = GPT(GPTConfig(n_layer=12, n_head=12, n_embd=768))
        self.model.load_pretrained()

        GPUS = ()
        if gpus > 1:
            GPUS = tuple(f'{Device.DEFAULT}:{i}' for i in range(gpus))
            for x in nn.state.get_parameters(self.model): 
                x.to_(GPUS)

        enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        self.decode = lambda l: enc.decode(l)

        batches_list = []
        i = 0
        while True:
            x = self.tokens[i:i+B*T].view(B, T)
            y = self.tokens[i+1:i+B*T+1].view(B, T)
            batches_list.append((x, y))
            i += B*T
            if i + B*T + 1 >= len(self.tokens):
                break

        x, y = batches_list[0]
        optimizer = nn.optim.AdamW(nn.state.get_parameters(self.model), lr=1e-4, weight_decay=0)

        print(f"model state:     {sum(x.nbytes() for x in nn.state.get_parameters(self.model))/1e9:.2f} GB")
        print(f"optimizer state: {sum(x.nbytes() for x in nn.state.get_parameters(optimizer))/1e9:.2f} GB")

        if gpus > 1: 
            x = x.shard(GPUS, axis=0)
            y = y.shard(GPUS, axis=0)

        @TinyJit
        @Tensor.train()
        def step(x:Tensor, y:Tensor) -> Tensor:
            _, loss = self.model(x, y)
            assert loss is not None
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
        
        self.next(self.end)
    
    @step
    def end(self):
        """Final step - report completion and optionally generate text using reference strategy."""
        print("Training completed successfully!")

        if not self.skip_test:
            try:
                # Check current device and attempt text generation
                sample_param = next(iter(nn.state.get_parameters(self.model)))
                current_device = str(sample_param.device)
                print(f"Model is currently on device: {current_device}")

                gpus = int(self.gpus)  # type: ignore

                if current_device.startswith('CUDA') and gpus > 0:
                    print("Attempting GPU text generation using reference strategy...")
                    try:
                        self._generate_text_reference_strategy()
                    except Exception as e:
                        print(f"GPU text generation failed: {e}")
                        print("Falling back to CPU for text generation...")
                        self._generate_text_cpu_reference()
                else:
                    print("Using reference strategy for text generation (CPU)...")
                    self._generate_text_cpu_reference()

            except Exception as e:
                print(f"Text generation failed: {e}")

        print("TinyGradLLMFlow is all done.")

    def _generate_text_reference_strategy(self):
        """Generate text using the reference implementation strategy."""
        print("Generating 16 tokens (reference strategy)...")

        # Initialize tokenizer for this process
        enc = tiktoken.get_encoding("gpt2")
        encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode_fn = lambda l: enc.decode(l)

        # Get current device
        sample_param = next(iter(nn.state.get_parameters(self.model)))
        
        print("Step 1: Moving all model parameters to current device...")
        for param in nn.state.get_parameters(self.model):
            if hasattr(param, 'to_'):
                param.to_(sample_param.device)

        print("Step 2: Synchronizing device...")
        Device[Device.DEFAULT].synchronize()
        
        print("Step 3: Setting up generation (reference style)...")
        start = "<|endoftext|>"
        start_ids = encode_fn(start)
        
        # This matches the reference exactly: (Tensor(start_ids)[None, ...])
        x = Tensor(start_ids)[None, ...]
        
        max_new_tokens = 16
        temperature = 1.0
        
        print(f"Input: '{start}' -> {len(start_ids)} tokens")
        
        print("Step 4: Generating text...")
        y = self.model.generate(x, max_new_tokens, temperature=temperature)
        
        print("Step 5: Extracting tokens (reference style)...")
        # This matches the reference exactly: y[0].tolist()
        output_tokens = y[0].tolist()
        
        print("Step 6: Decoding and displaying...")
        decoded_text = decode_fn(output_tokens)
        
        print(f"✅ SUCCESS! Generated text: '{decoded_text}'")
        print(f"Total tokens generated: {len(output_tokens)}")
        
        if len(output_tokens) > 1:
            new_tokens = output_tokens[1:]  # Skip the initial endoftext token
            print(f"New tokens: {new_tokens}")

    def _generate_text_cpu_reference(self):
        """Generate text using CPU with reference strategy."""
        print("Generating 16 tokens on CPU (reference strategy)...")

        # Initialize tokenizer for this process
        enc = tiktoken.get_encoding("gpt2")
        encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode_fn = lambda l: enc.decode(l)

        print("Step 1: Moving all model parameters to default device...")
        for param in nn.state.get_parameters(self.model):
            if hasattr(param, 'to_'):
                param.to_(Device.DEFAULT)

        print("Step 2: Synchronizing...")
        Device[Device.DEFAULT].synchronize()
        
        print("Step 3: Generating text...")
        start = "<|endoftext|>"
        start_ids = encode_fn(start)
        x = Tensor(start_ids)[None, ...]
        
        max_new_tokens = 16
        temperature = 1.0
        
        y = self.model.generate(x, max_new_tokens, temperature=temperature)
        
        # Simple .tolist() extraction like reference
        output_tokens = y[0].tolist()
        
        decoded_text = decode_fn(output_tokens)
        print(f"✅ SUCCESS! Generated text: '{decoded_text}'")
        print(f"Total tokens generated: {len(output_tokens)}")

if __name__ == "__main__":
    TinyGradLLMFlow()