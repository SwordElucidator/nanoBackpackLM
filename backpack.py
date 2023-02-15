from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np

from model import Block, LayerNorm


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
        if not bool(torch.all(torch.abs(debug_m.data - o.data) < 1e-5).numpy()):
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
        return torch.softmax(self.E(o), dim=-1)  # batch, n, vocab


class BackpackLM(nn.Module):
    def __init__(self, config: BackpackLMConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = Transformer(config)
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

        # o: (d, n)    oj:  (d, 1)
        self.backpack = Backpack(
            sense_func, contextualization_weight_func,
            lambda o: o @ self.transformer.wte.weight.T,  # (vocab, d)   o: (n, d)
        )

    def forward(self, idx):
        return self.backpack(idx)


# vocab: 10   d: 3     n: 5     batch: 16  k: 7
if __name__ == '__main__':
    # test
    lm = BackpackLM(BackpackLMConfig(
        block_size=11, vocab_size=10, n_layer=6, n_head=3, n_embd=21, n_sense_vector=7
    ))
    x = torch.zeros((16, 8), dtype=torch.int)
    rst = lm(x)
    print(rst)
    print(rst.shape)
