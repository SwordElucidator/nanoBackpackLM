import os

import numpy as np
import torch
from transformers import BertTokenizer, AutoTokenizer

HUGGINGFACE_TOKENIZERS = {
    'chinese-character-bert': (BertTokenizer, "uer/gpt2-chinese-cluecorpussmall"),
    'xlm-250k': (AutoTokenizer, 'xlm-roberta-base')
}


def get_batch_function_for_multilingual_training(dataset, data_bin_dtype, alpha, block_size, batch_size, device):
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    data_dir = os.path.join('data', dataset)
    langs = [n.split('_train.bin')[0] for n in os.listdir(data_dir) if n.endswith('_train.bin')]
    dic = {
        lang: {
            'train': np.memmap(os.path.join(data_dir, f'{lang}_train.bin'), dtype=getattr(np, data_bin_dtype), mode='r'),
            'val': np.memmap(os.path.join(data_dir, f'{lang}_val.bin'), dtype=getattr(np, data_bin_dtype), mode='r')
        } for lang in langs
    }
    n_dict = {}
    for lang, v in dic.items():
        n_dict[lang] = len(v['train'])
    p_dict = {lang: n / sum(n_dict.values()) for lang, n in n_dict.items()}
    q_dict = {lang: p ** alpha / sum(pp ** alpha for pp in p_dict.values()) for lang, p in p_dict.items()}

    def get_batch(split):
        lang = np.random.choice(list(q_dict.keys()), p=list(q_dict.values()))
        data = dic[lang]['train'] if split == 'train' else dic[lang]['val']
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    return get_batch


if __name__ == "__main__":
    get_batch = get_batch_function_for_multilingual_training('common_crawl', 'uint32', 0.3)
