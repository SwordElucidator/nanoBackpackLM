import inspect
import math
from abc import ABC
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import Block, LayerNorm, new_gelu

DEBUGGING = True


# Backpack Prototype
class SenseVectorLayer(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def sense_func(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.sense_func(x)  # (n, d, k)


class ContextualizationLayer(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def contextualization_weight_func(self, x):
        raise NotImplementedError

    def __forward_test(self, alpha, sense_x):  # for debug    batch, n, d, k
        o = torch.zeros((sense_x.shape[0], sense_x.shape[1], sense_x.shape[2]))  # (batch, n, d)
        for i in range(sense_x.shape[1]):   # n
            for j in range(sense_x.shape[1]):  # n
                for l in range(alpha.shape[1]):  # k
                    for b in range(alpha.shape[0]):
                        o[b, i, :] += alpha[b, l, i, j] * sense_x[b, j, :, l]
        return o  # (batch, n, d)

    def forward(self, x, sense_x):
        alpha = self.contextualization_weight_func(x)  # (k, n, n)  for k sense vectors (weights) of nxn matrix
        o = torch.sum(torch.matmul(alpha, sense_x.permute(0, 3, 1, 2)), dim=1)  # (n, d)   from  (k, n, n)   (n, d, k)
        if DEBUGGING:
            assert bool(torch.all(torch.abs(self.__forward_test(alpha, sense_x).data - o.data) < 1e-5).numpy())
        return o  # (batch, n, d)


class LogitLayer(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def logit_func(self, o):
        raise NotImplementedError

    def forward(self, o):
        return self.logit_func(o)


class Backpack(nn.Module):
    def __init__(self, sense_vector_layer: SenseVectorLayer,
                 contextualization_layer: ContextualizationLayer, logit_layer: LogitLayer):
        super().__init__()
        self.logit_layer = logit_layer  # (n, d) -> |y|
        self.sense_vector_layer = sense_vector_layer
        self.contextualization_layer = contextualization_layer

    def forward(self, x):
        sense_x = self.sense_vector_layer(x)  # batch, n, d, k
        o = self.contextualization_layer(x, sense_x)  # (batch, n, d)
        return self.logit_layer(o)  # no softmax


# BackpackLM
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


class BackpackFF(nn.Module):

    def __init__(self, from_dim, to_dim, dropout, bias):
        super().__init__()
        self.c_fc = nn.Linear(from_dim, 4 * from_dim, bias=bias)
        self.c_proj = nn.Linear(4 * from_dim, to_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LMSenseVectorLayer(SenseVectorLayer):

    def __init__(self, wte, config: BackpackLMConfig):
        super().__init__()
        self.wte = wte
        self.n_embd, self.n_sense_vector = config.n_embd, config.n_sense_vector
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ff_1 = BackpackFF(config.n_embd, config.n_embd, config.dropout, config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ff_2 = BackpackFF(config.n_embd, config.n_embd * config.n_sense_vector, config.dropout, config.bias)

    def sense_func(self, x):
        x = self.wte(x)
        x = x + self.ff_1(self.ln_1(x))  # TODO before or after?
        # TODO addtional res-net here?
        # x = x.unsqueeze(dim=-1) + self.ff_2(self.ln_2(x)).reshape(*x.shape, self.n_sense_vector)  # put res net to all k sense vectors of the result
        x = self.ff_2(self.ln_2(x)).reshape(*x.shape, self.n_sense_vector)
        return x


class LMContextualizationLayer(ContextualizationLayer):

    def __init__(self, wte, config):
        super().__init__()
        self.n_sense_vector, self.n_embd = config.n_sense_vector, config.n_embd
        self.transformer = Transformer(config)
        self.transformer.wte = wte
        self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=False)

    def contextualization_weight_func(self, x):
        h = self.transformer(x)  # (batch, n, d)
        batch_size, seq_len, emb_dim = h.shape
        q, k = self.c_attn(h).split(self.n_embd, dim=2)
        k = k.view(batch_size, seq_len, self.n_sense_vector, emb_dim // self.n_sense_vector).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.n_sense_vector, emb_dim // self.n_sense_vector).transpose(1, 2)
        att = (q @ k.transpose(-2, -1))
        return torch.softmax(att, dim=-1)


class LMLogitLayer(LogitLayer):
    def __init__(self, wte, config: BackpackLMConfig):
        super(LMLogitLayer, self).__init__()
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

    def logit_func(self, o):
        return self.lm_head(o)


class BackpackLM(nn.Module):
    def __init__(self, config: BackpackLMConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        self.backpack = Backpack(
            LMSenseVectorLayer(self.wte, config),
            LMContextualizationLayer(self.wte, config),
            LMLogitLayer(self.wte, config)
        )

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

    def _init_weights(self, module):  # copied from gpt2
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
