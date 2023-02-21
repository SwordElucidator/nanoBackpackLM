"""
this file includes several experiments on the sense vector
"""
import pickle

import torch

from experiments.load_model import load_model
from experiments.utils import TopK


@torch.no_grad()
class SenseVectorExperiment(object):
    def __init__(self):
        self.model, self.encode, self.decode = load_model()
        self.sense_vector = self.model.sense_vector(mini_batch_size=4)
        self.vocab_size = self.sense_vector.shape[0]
        self.n_sense_vectors = self.sense_vector.shape[1]

    @torch.no_grad()
    def cosine_similarity(self, x1id, x2id):
        return torch.cosine_similarity(self.sense_vector[x1id], self.sense_vector[x2id], dim=-1)

    @torch.no_grad()
    def min_sense_cosine(self, x1id, x2id):
        return torch.min(self.cosine_similarity(x1id, x2id))

    @torch.no_grad()
    def min_sense_cosine_matrix(self, words, k=10):
        sense_dict = {word: [TopK(k) for _ in range(self.n_sense_vectors)] for word in words}
        id2word = {k: self.decode([k]) for k in range(0, self.vocab_size)}
        word2id = {word: self.encode(word)[1] for word in words}
        for word in words:
            print(f'analysing on word {word}')
            similarities = self.cosine_similarity(word2id[word], torch.arange(0, self.vocab_size))  # len, k
            for j in range(0, self.vocab_size):
                if j == word2id[word]:
                    continue
                for l, v in enumerate(similarities[j]):
                    sense_dict[word][l].append((float(v), id2word[j]))
        return sense_dict


if __name__ == '__main__':
    ex = SenseVectorExperiment()
    words = ['天', '地', '沙', '哲']
    cos_sim = ex.min_sense_cosine_matrix(words)
    pickle.dump(cos_sim, open('sense_vector.p', 'wb+'))
    for word in words:
        print(word)
        for i in range(ex.n_sense_vectors):
            top_k = cos_sim[word][i].top_k()
            print(f"sense {i}: " + ' '.join(c[1] for c in top_k))
