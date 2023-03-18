import json

import torch
from transformers import GPT2LMHeadModel

from experiments.loader import load_model, device
from torch.nn import functional as F


def load_dev_set():
    dev_set_path = 'data/wcpc/dev.json'
    with open(dev_set_path, 'r') as f:
        return [json.loads(line) for line in f.readlines()]


def chinese_wcpc_test(huggingface_gpt=False, k=3):
    top1, top3 = 0, 0
    model, encode, decode = load_model()
    if huggingface_gpt:
        model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    dev_set = load_dev_set()
    for data in dev_set:
        text, true_word = data['masked_text'], data['correct_word']
        start = text[:text.index('<mask>')]
        length = text.count('<mask>')
        start_tokens = encode(start)[:- 1]
        beam_search = [(torch.tensor(encode(start)[:- 1], dtype=torch.long, device=device), 1)]
        while length:
            new_beam = []
            for tokens, prob in beam_search:
                logits = model(tokens[None, ...])[0][0, -1, :]
                topk = torch.topk(F.softmax(logits, dim=-1), k=k + 2)
                for index, new_prob in zip(topk.indices, topk.values):
                    if decode(index) not in [
                        ',', '，', '。', '[UNK]', '！', '!', '；', ';', '?', '？', '[ U N K ]', '"', '”', '“', '、',
                        '：', ':', '「', '」', '【', '】', '`', '…', '……'
                    ] and index != 100:
                        new_beam.append((torch.cat((tokens, torch.tensor([index]))), new_prob * prob))
            length -= 1
            beam_search = sorted(new_beam, key=lambda x: -x[1])[:k]
        print(f'right word: {true_word}')
        for i, (tokens, _) in enumerate(beam_search):
            word = decode(tokens[len(start_tokens):]).replace(' ', '')
            if not word.strip():
                import pdb
                pdb.set_trace()
            print(f'guess: {word}')
            if word == true_word:
                if i == 0:
                    top1 += 1
                top3 += 1
                break
        print()
    return top1 / len(dev_set), top3 / len(dev_set)


def chinese_wcpc_test_v2(huggingface_gpt=False, k=3, max_holding_beam=10):
    top1, top3 = 0, 0
    model, encode, decode = load_model()
    if huggingface_gpt:
        model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    dev_set = load_dev_set()
    for data in dev_set:
        text, true_word = data['masked_text'], data['correct_word']
        start = text[:text.index('<mask>')]
        length = text.count('<mask>')
        start_tokens = encode(start)[:- 1]
        beam_search = [(torch.tensor(encode(start)[:- 1], dtype=torch.long, device=device), 0)]
        while length:
            new_beam = []
            for tokens, log_prob in beam_search:
                logits = model(tokens[None, ...])[0][0, -1, :]
                topk = torch.topk(F.softmax(logits, dim=-1), k=max_holding_beam * 3)
                for index, new_log_prob in zip(topk.indices, torch.log(topk.values)):
                    if decode(index) not in [
                        ',', '，', '。', '[UNK]', '！', '!', '；', ';', '?', '？', '[ U N K ]', '"', '”', '“', '、',
                        '：', ':', '「', '」', '【', '】', '`', '…', '……'
                    ] and index != 100:
                        new_beam.append((torch.cat((tokens, torch.tensor([index]))), new_log_prob + log_prob))
            length -= 1
            beam_search = sorted(new_beam, key=lambda x: -x[1])[:max_holding_beam]

        # last words punishment
        last_words = text[len(start) + text.count('<mask>') * 6:]
        for i in range(len(last_words)):
            additional_tokens = torch.tensor(encode(last_words[:i])[1: -1], dtype=torch.int64)
            new_beam = []
            for tokens, log_prob in beam_search:
                logits = model(torch.cat((tokens, additional_tokens))[None, ...])[0][0, -1, :]
                log_expect = torch.log(logits[encode(last_words[i])[1: -1]])
                new_beam.append((tokens, log_expect + log_prob))
            beam_search = sorted(new_beam, key=lambda x: -x[1])

        print(f'right word: {true_word}')
        for i, (tokens, _) in enumerate(beam_search[:k]):
            word = decode(tokens[len(start_tokens):]).replace(' ', '')
            if not word.strip():
                import pdb
                pdb.set_trace()
            print(f'guess: {word}')
            if word == true_word:
                if i == 0:
                    top1 += 1
                top3 += 1
                break
        print()
    return top1 / len(dev_set), top3 / len(dev_set)


if __name__ == '__main__':
    print(chinese_wcpc_test(huggingface_gpt=False))
