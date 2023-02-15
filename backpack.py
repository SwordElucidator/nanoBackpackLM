import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import Block, LayerNorm


DEBUGGING = True


class SenseVectorLayer(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C

    def forward(self, x):
        return self.C(x)  # n, d, k


class ContextualizationLayer(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A

    def __sum_test(self, alpha, C_x):  # for debug    batch, n, d, k
        o = torch.zeros((C_x.shape[0], C_x.shape[1], C_x.shape[2]))  # (batch, n, d)
        for i in range(C_x.shape[1]):   # n
            for j in range(C_x.shape[1]):  # n
                for l in range(alpha.shape[1]):  # k
                    for b in range(alpha.shape[0]):
                        o[b, i, :] += alpha[b, l, i, j] * C_x[b, j, :, l]
        return o  # (batch, n, d)

    def forward(self, x, C_x):
        alpha = self.A(x)  # (k, n, n)  for k sense vectors (weights) of nxn matrix
        o = torch.sum(torch.matmul(alpha, C_x.permute(0, 3, 1, 2)), dim=1)  # (n, d)   from  (k, n, n)   (n, d, k)
        debug_m = self.__sum_test(alpha, C_x)
        if DEBUGGING and not bool(torch.all(torch.abs(debug_m.data - o.data) < 1e-5).numpy()):
            raise ValueError
        return o  # (batch, n, d)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x


@dataclass
class BackpackLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    n_sense_vector: int = 16


class Backpack(nn.Module):
    def __init__(self, C, A, E):
        super().__init__()
        self.E = E  # (n, d) -> |y|
        self.sense_vector_layer = SenseVectorLayer(C)
        self.contextualization_layer = ContextualizationLayer(A)

    def forward(self, x):
        C_x = self.sense_vector_layer(x)  # batch, n, d, k
        o = self.contextualization_layer(x, C_x)  # (batch, n, d)
        return self.E(o)  # no softmax


class BackpackLM(nn.Module):
    def __init__(self, config: BackpackLMConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        self.FF = nn.Linear(config.n_embd, config.n_embd * self.config.n_sense_vector)
        self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=False)

        def sense_func(x):  # n, d  => n, d, k
            return self.FF(self.transformer.wte(x)).reshape(*x.shape, config.n_embd, config.n_sense_vector)

        def contextualization_weight_func(x):
            h = self.transformer(x)  # (batch, n, d)
            batch_size, seq_len, emb_dim = h.shape
            q, k = self.c_attn(h).split(self.config.n_embd, dim=2)
            k = k.view(batch_size, seq_len, config.n_sense_vector, emb_dim // config.n_sense_vector).transpose(1, 2)
            q = q.view(batch_size, seq_len, config.n_sense_vector, emb_dim // config.n_sense_vector).transpose(1, 2)
            att = (q @ k.transpose(-2, -1))
            return torch.softmax(att, dim=-1)

        def logit_func(o):
            return self.lm_head(o)

        # o: (d, n)    oj:  (d, 1)
        self.backpack = Backpack(sense_func, contextualization_weight_func, logit_func)

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, idx, targets=None):
        logits = self.backpack(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):  # same optimizer with gpt2
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer


# vocab: 10   d: 3     n: 5     batch: 16  k: 7
if __name__ == '__main__':
    # test
    lm = BackpackLM(BackpackLMConfig(
        block_size=11, vocab_size=10, n_layer=6, n_head=3, n_embd=21, n_sense_vector=7
    ))
    x = torch.zeros((1, 8), dtype=torch.int)
    logits, loss = lm(x, torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]))
    print(F.softmax(logits, dim=-1))
    print(logits.shape)
    print(f'loss: {loss}')
