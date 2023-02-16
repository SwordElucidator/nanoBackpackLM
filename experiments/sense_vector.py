"""
this file includes several experiments on the sense vector
"""
from collections import defaultdict

import torch

from experiments.load_model import load_model
from experiments.utils import TopK


@torch.no_grad()
class SenseVectorExperiment(object):
    def __init__(self):
        self.model, self.ecode, self.decode = load_model()
        self.sense_vector = self.model.sense_vector()
        self.vocab_size = self.sense_vector.shape[0]
        self.n_sense_vectors = self.sense_vector.shape[1]

    def cosine_similarity(self, x1id, x2id):
        return torch.cosine_similarity(self.sense_vector[x1id], self.sense_vector[x2id], dim=-1)

    def min_sense_cosine(self, x1id, x2id):
        return torch.min(self.cosine_similarity(x1id, x2id))
    
    def min_sense_cosine_matrix(self, k=5):
        sense_dict = [defaultdict(lambda: TopK(k)) for _ in range(self.n_sense_vectors)]
        for i in range(self.vocab_size):
            for j in range(i + 1, self.vocab_size):
                word_i, word_j = self.decode([i]), self.decode([j])
                for l, v in enumerate(self.cosine_similarity(i, j)):
                    sense_dict[l][word_i].append((v, word_j))
                    sense_dict[l][word_j].append((v, word_i))
        return sense_dict


if __name__ == '__main__':
    ex = SenseVectorExperiment()
    cos_sim = ex.min_sense_cosine_matrix()
    print(cos_sim)
