"""
this file includes several experiments on the sense vector
"""
import pickle

import numpy as np
import torch

from experiments.loader import load_model, device
from experiments.utils import TopK


@torch.no_grad()
class SenseVectorExperiment(object):
    def __init__(self):
        self.model, self.encode, self.decode = load_model()
        self.sense_vector = self.model.sense_vectors(device=device)
        self.vocab_size = self.sense_vector.shape[0]
        self.word_vectors = self.model.wte(torch.arange(self.vocab_size).to(device))
        self.n_sense_vectors = self.sense_vector.shape[1]
        self.id2word = {k: self.decode([k]) for k in range(0, self.vocab_size)}
        self.word2id = {word: self.encode(word)[1] for word in self.id2word.values()}

    @torch.no_grad()
    def sense_projection(self, word, k=5):
        senses = self.sense_vector[self.word2id[word]].to(device)
        output = self.model.backpack.logit_layer(senses)
        topk = torch.topk(output, 5, dim=-1).indices.to('cpu').numpy()
        return [[self.id2word[i] for i in row] for row in topk]

    @torch.no_grad()
    def cosine_similarity(self, x1id, x2id):
        return torch.cosine_similarity(self.sense_vector[x1id], self.sense_vector[x2id], dim=-1)

    @torch.no_grad()
    def min_sense_cosine(self, x1id, x2id):
        return torch.min(self.cosine_similarity(x1id, x2id))

    @torch.no_grad()
    def min_sense_cosine_matrix(self, words, k=5):
        sense_dict = {word: [TopK(k) for _ in range(self.n_sense_vectors)] for word in words}
        for word in words:
            print(f'analysing on word {word}')
            similarities = self.cosine_similarity(self.word2id[word], torch.arange(0, self.vocab_size))  # len, k
            for j in range(0, self.vocab_size):
                if j == self.word2id[word]:
                    continue
                for l, v in enumerate(similarities[j]):
                    sense_dict[word][l].append((float(v), self.id2word[j]))
        return sense_dict


if __name__ == '__main__':
    ex = SenseVectorExperiment()
    words = ['天', '地', '沙', '哲']
    for word in words:
        print(ex.sense_projection(word))
    # cos_sim = ex.min_sense_cosine_matrix(words)
    # pickle.dump(cos_sim, open('sense_vector.p', 'wb+'))
    # for word in words:
    #     print(word)
    #     for i in range(ex.n_sense_vectors):
    #         top_k = cos_sim[word][i].top_k()
    #         print(f"sense {i}: " + ' '.join(c[1] for c in top_k))
