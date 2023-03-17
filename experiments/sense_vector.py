"""
this file includes several experiments on the sense vector
"""
import operator
from functools import reduce

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from experiments.loader import load_model, device
from experiments.utils import TopK
from model import GPT
from torch.nn import functional as F


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
        topk = torch.topk(output, k, dim=-1).indices.to('cpu').numpy()
        return [[self.id2word[i] for i in row] for row in topk]

    @torch.no_grad()
    def compositional_sense_projection(self, words, strategy='avg'):
        if strategy == 'avg':
            sense = reduce(operator.add, (self.sense_vector[self.word2id[word]] for word in words)) / len(words)
        elif strategy == 'contextualized':
            sense = self.get_contextualized_sense(words)
        else:
            raise NotImplementedError
        output = self.model.backpack.logit_layer(sense)
        return output

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

    @torch.no_grad()
    def cosine_similarity_on_chinese_characters(self, word1, word2=None, strategy='avg'):
        if strategy == 'avg':
            word1_sense = reduce(operator.add, (self.sense_vector[self.word2id[c]] for c in word1)) / len(word1)
            if word2:
                word2_sense = reduce(operator.add, (self.sense_vector[self.word2id[c]] for c in word2)) / len(word2)
            else:
                word2_sense = self.sense_vector[torch.arange(0, self.vocab_size)]
        elif strategy == 'contextualized':
            word1_sense = self.get_contextualized_sense(word1)
            if word2:
                word2_sense = self.get_contextualized_sense(word2)
            else:
                word2_sense = self.sense_vector[torch.arange(0, self.vocab_size)]
        else:
            raise NotImplementedError
        return torch.cosine_similarity(word1_sense, word2_sense, dim=-1)

    @torch.no_grad()
    def min_sense_cosine_matrix_on_chinese_characters(self, words, strategy='avg', k=5):
        sense_dict = {word: [TopK(k) for _ in range(self.n_sense_vectors)] for word in words}
        for word in words:
            print(f'analysing on word {word}')
            similarities = self.cosine_similarity_on_chinese_characters(word, strategy=strategy)
            for j in range(0, self.vocab_size):
                existed = False
                for character in word:
                    if j == self.word2id[character]:
                        existed = True
                        break
                if existed:
                    continue
                for l, v in enumerate(similarities[j]):
                    sense_dict[word][l].append((float(v), self.id2word[j]))
        return sense_dict

    def get_contextualized_sense(self, words):
        contextualized_sense = self.model.backpack.analyse_sense_contextualization(
            torch.tensor(self.encode(words)[1:- 1], dtype=torch.long, device=device).unsqueeze(0)
        )
        new_sense = contextualized_sense[-1, -1, :, :]
        return new_sense

    def generate_with(self, words, length=10):
        start_words = torch.tensor(self.encode(words)[1:- 1], dtype=torch.long, device=device)[None, ...]
        return self.decode(self.model.generate(start_words, length)[0].tolist())

    def next_topk_words(self, words, k=5):
        start_words = torch.tensor(self.encode(words)[1:- 1], dtype=torch.long, device=device)[None, ...]
        return torch.topk(self.model(start_words)[0][0, -1, :], k=k)

    def word_composition(self, word, pieces):
        out = {}
        for piece in pieces:
            alpha = self.model.backpack.contextualization_layer.contextualization_weight_func(
                torch.tensor(self.encode(piece)[1:- 1], dtype=torch.long, device=device).unsqueeze(0)
            ).squeeze()
            idx = piece.index(word)
            related_alpha = alpha[:, :, idx: idx + len(word)]
            composition_ratio = related_alpha / torch.sum(related_alpha, axis=2).unsqueeze(-1)
            usable_composition_ratio = composition_ratio[:, composition_ratio[0, :, -1] > 0]
            out[piece] = usable_composition_ratio  # n_sense_vector, usable_sequence_length, len(word)~ratio
        return out

    def chinese_word_sim_240_297_test(self):  # for sense vector
        word_sim_type = ['240', '297']
        for x in word_sim_type:
            file = open('data/similarity/wordsim-' + x + '.txt')

            word_sim_std = []
            word_sim_pre = {i: [] for i in range(self.model.config.n_sense_vector)}
            for line in file:
                word1, word2, val_str = line.strip().split()

                # sense1 = self.get_contextualized_sense(word1)
                # sense2 = self.get_contextualized_sense(word2)
                encoded1, encoded2 = self.encode(word1)[1:-1], self.encode(word2)[1:-1]
                sense1 = reduce(operator.add, (self.sense_vector[c] for c in encoded1)) / len(encoded1)
                sense2 = reduce(operator.add, (self.sense_vector[c] for c in encoded2)) / len(encoded2)

                word_sim_std.append(float(val_str))
                # cos_sim = np.dot(sense1, sense2)
                cos_sim = torch.sum(sense1 * sense2, -1).detach().numpy()
                for i, sim in enumerate(cos_sim):
                    word_sim_pre[i].append(sim)
            for i in range(self.model.config.n_sense_vector):
                corr_coef = np.corrcoef(word_sim_std, word_sim_pre[i])[0, 1]
                print(f'WordSim-{x} sense:{i + 1} Score:{corr_coef}')
            file.close()

    def _get_huggingface_wte_layer(self, name):
        model = GPT2LMHeadModel.from_pretrained(name)  # "uer/gpt2-chinese-cluecorpussmall"
        return model.transformer.wte

    def _get_local_wte_layer(self, name):
        model = load_model(GPT, name)[0]
        return model.transformer.wte

    def chinese_word_sim_240_297_test_on_gpt2(self, name='clue_micro_gpt2', from_huggingface=False):
        if from_huggingface:
            wte_layer = self._get_huggingface_wte_layer(name)
        else:
            wte_layer = self._get_local_wte_layer(name)
        word_sim_type = ['240', '297']
        for x in word_sim_type:
            file = open('data/similarity/wordsim-' + x + '.txt')

            word_sim_std = []
            word_sim_pre = []
            for line in file:
                word1, word2, val_str = line.strip().split()
                encoded1, encoded2 = self.encode(word1)[1:-1], self.encode(word2)[1:-1]
                sense1 = torch.sum(wte_layer(torch.tensor(encoded1)), dim=0) / len(encoded1)
                sense2 = torch.sum(wte_layer(torch.tensor(encoded2)), dim=0) / len(encoded2)

                word_sim_std.append(float(val_str))
                # cos_sim = np.dot(sense1, sense2)
                cos_sim = float(torch.sum(sense1 * sense2, -1).detach().numpy())
                word_sim_pre.append(cos_sim)
            corr_coef = np.corrcoef(word_sim_std, word_sim_pre)[0, 1]
            print(f'WordSim-{x} Score:{corr_coef}')
            file.close()

    # def _beam_search
    def alpha_modification_generation(self, prompt, length, alpha_modifier=None, k=3):
        # patch
        original_weight_func = self.model.backpack.contextualization_layer.contextualization_weight_func

        if alpha_modifier:
            def new_weight_func(x):
                alpha = original_weight_func(x)
                return alpha_modifier(alpha)

            self.model.backpack.contextualization_layer.contextualization_weight_func = new_weight_func

        start_tokens = self.encode(prompt)[:- 1]
        beam_search = [(torch.tensor(start_tokens, dtype=torch.long, device=device), 1)]
        left_length = length
        while left_length:
            new_beam = []
            for tokens, prob in beam_search:

                logits = self.model(tokens[None, ...])[0][0, -1, :]
                topk = torch.topk(F.softmax(logits, dim=-1), k=k + 2)
                for index, new_prob in zip(topk.indices, topk.values):
                    if self.decode(index) not in [
                        '[UNK]', '；', ';', '[ U N K ]', '"', '”', '“', '、', '「', '」', '【', '】', '`', '…', '……'
                    ] and index != 100:
                        new_beam.append((torch.cat((tokens, torch.tensor([index]))), new_prob * prob))
                    new_beam.append((torch.cat((tokens, torch.tensor([index]))), new_prob * prob))
            left_length -= 1
            beam_search = sorted(new_beam, key=lambda x: -x[1])[:k]

        self.model.backpack.contextualization_layer.contextualization_weight_func = original_weight_func
        return [(self.decode(tokens), prob) for tokens, prob in beam_search]

    def amplified_generation_test(self, prompt, length, target_word, amplifier, k=3):
        start_index = prompt.index(target_word) + 1

        def modifier(alpha):
            amp = torch.tensor(amplifier)
            new_ratio = amp[None, None, :].expand(1, alpha.shape[1], len(amp))
            original_weight_sum = alpha[:, :, -1, start_index: start_index + len(target_word)].sum(dim=-1)
            new_weight_sum = alpha[:, :, -1, start_index: start_index + len(target_word)] * new_ratio
            new_weight = new_weight_sum / new_weight_sum.sum(-1).unsqueeze(2) * original_weight_sum.unsqueeze(2)
            alpha[:, :, -1, start_index: start_index + len(target_word)] = new_weight
            return alpha

        return self.alpha_modification_generation(prompt, length, modifier, k)

    def bias_check_on_sense_vector(self, words, target_words):
        logits = self.compositional_sense_projection(words, 'contextualized')
        log_probs = torch.log(F.softmax(logits, dim=-1))
        probs = []
        for target_word in target_words:
            assert len(target_word) == 1
            word_id = self.encode(target_word)[1]
            probs.append(log_probs[:, word_id])
        return probs

    def debiasing_test_on_generate_target_words(self, prompts, to_modify_word, bias_words, sense_index=10, sense_ratio=0.5):
        """
        some prompts
        """

        old_sense_vector_layer_forward = self.model.backpack.sense_vector_layer.forward
        assert len(bias_words) == 2
        bias_indices = self.encode(bias_words)[1: -1]
        to_modify_indices = self.encode(to_modify_word)[1: -1]

        def modified_sense_vector(x):
            sense_vec = old_sense_vector_layer_forward(x)
            word_indices = [i for i, ind in enumerate(x[0]) if ind in to_modify_indices]
            sense_vec[0, word_indices, :, sense_index] *= sense_ratio
            return sense_vec

        try:
            self.model.backpack.sense_vector_layer.forward = modified_sense_vector

            total_diff = 0
            for prompt in prompts:
                tokens = torch.tensor(self.encode(prompt)[:- 1])
                logits = self.model(tokens[None, ...])[0][0, -1, :]
                probs = F.softmax(logits, dim=-1)
                diff = torch.max(probs[bias_indices[0]] / probs[bias_indices[1]],
                                 probs[bias_indices[1]] / probs[bias_indices[0]])
                total_diff += diff
        finally:
            self.model.backpack.sense_vector_layer.forward = old_sense_vector_layer_forward
        return total_diff / len(prompts)


if __name__ == '__main__':
    ex = SenseVectorExperiment()
    for word in ['兵', '警', '官', '僧', '尼', '师', '牧', '护', '洁']:
        print(word)
        he, her = ex.bias_check_on_sense_vector(word, ['他', '她'])
        # max_diff = torch.argmax(torch.abs(he - her))
        print((he - her)[3])
    characters = ['兵', '警', '官', '僧', '尼', '师', '牧', '护', '保洁']
    combined_words = ['士兵', '警察', '官员', '僧人', '老尼', '老师', '牧民', '护士', '保洁']
    prompts = [
        '我的%s说，',
        '这个%s相信',
        '%s进到屋子里，',
    ]
    for word, combined_word in zip(characters, combined_words):
        print(word)
        for i in range(16):
            if i == 3:
                print(f'sense {i}')
                for r in [1, 0.6, 0.3, 0]:
                    print(r, ex.debiasing_test_on_generate_target_words(
                        [p % combined_word for p in prompts], word, ['他', '她'], i, r
                    ))
            # print()
    # print(ex.alpha_modification_generation('光污染', 20, None))
    # print(ex.amplified_generation_test('光污染', 20, '光污染', [4, 1, 1]))
    # print(ex.amplified_generation_test('光污染', 20, '光污染', [1, 2, 2]))

    # ex.chinese_word_sim_240_297_test()
    # ex.chinese_word_sim_240_297_test_on_gpt2()
    # word_composition = ex.word_composition('新闻', [
    #     '新闻',
    #     '最近新闻里常报道特朗普总统的一些逸闻，有时也让人哭笑无泪。', '新闻时间即将结束，接下来是动画片放映时间',
    #     '有篇新闻，你要不看看。相信我，多读读总有好处的',
    #     '每天都有很多新闻从世界的四面八方向本报社汇聚'
    # ])
    # # reduce(operator.add, (torch.sum(compos, dim=1) / compos.shape[1] for compos in word_composition.values())) / len(
    #     # word_composition)
    # base = None
    # for key, compos in word_composition.items():
    #     print(key)
    #     # print(ex.decode(ex.next_topk_words(key, 5).indices.detach().numpy()))
    #     avg_compos = torch.sum(compos, dim=1) / compos.shape[1]
    #     if base is None:
    #         base = avg_compos
    #     print(torch.abs(avg_compos - base))
    # print(base)  # n_sense, words_len:  basically

    # words = ['天', '地', '沙', '哲', '政治', '既然', '合理性', '网络故', '画蛇添', '政治因']
    # words = ['网络故', '突发网络故', '有一段网络故']

    # words = ['画蛇添']
    words = ['他', '她', '男', '女']
    # for word in words:
    #     print(ex.sense_projection(word))
    # for word in words:
    #     print(f'———————————————————— {word} ————————————————————')
    #     next_topk = ex.next_topk_words(word, 10)
    #     # print(ex.decode(next_topk.indices.to('cpu')))
    #     sense_projection = ex.compositional_sense_projection(word, 'contextualized')
    #     topk_vals = next_topk.values.to('cpu').detach().numpy()
    #     # for i, index in enumerate(next_topk.indices.to('cpu').numpy()):
    #     #     print(ex.id2word[index], topk_vals[i])
    #     #     print(sense_projection[:, index])
    #     topk = torch.topk(sense_projection, 10, dim=-1).indices.to('cpu').numpy()
    #     print([[ex.id2word[i] for i in row] for row in topk])

    # print(ex.cosine_similarity_on_chinese_characters('足球', '足球'))
    # print(ex.cosine_similarity_on_chinese_characters('入场券', '门票'))
    # print(ex.cosine_similarity_on_chinese_characters('钱', '现金'))
    # print(ex.cosine_similarity_on_chinese_characters('心理学', '抑郁'))
    # print(ex.cosine_similarity_on_chinese_characters('天', '日', 'contextualized'))
    # print(ex.cosine_similarity_on_chinese_characters('天', '日'))
    # print(ex.cosine_similarity_on_chinese_characters('水', '液', 'contextualized'))
    # print(ex.cosine_similarity_on_chinese_characters('壁', '墙', 'contextualized'))
    # print(ex.cosine_similarity_on_chinese_characters('足球', '足球', 'contextualized'))
    # print(ex.cosine_similarity_on_chinese_characters('入场券', '门票', 'contextualized'))
    # print(ex.cosine_similarity_on_chinese_characters('入场券', '门票'))
    # print(ex.cosine_similarity_on_chinese_characters('钱', '钞'))
    # print(ex.cosine_similarity_on_chinese_characters('钱', '现金', 'contextualized'))
    # print(ex.cosine_similarity_on_chinese_characters('心理学', '抑郁', 'contextualized'))
    # words = ['钱']
    # cos_sim_simple = ex.min_sense_cosine_matrix_on_chinese_characters(words)
    # # cos_sim = ex.min_sense_cosine_matrix_on_chinese_characters(words, strategy='contextualized')
    # for word in words:
    #     print(word)
    #     print(ex.generate_with(word, 20))
    #     for i in range(ex.n_sense_vectors):
    #         top_k_simple = cos_sim_simple[word][i].top_k()
    #         print(f"additional sense {i}: " + ' '.join(c[1] for c in top_k_simple))
    #         # top_k = cos_sim[word][i].top_k()
    #         # print(f"contextualized sense {i}: " + ' '.join(c[1] for c in top_k))
