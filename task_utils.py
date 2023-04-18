import os

import numpy as np
import torch
from transformers import BertTokenizer, AutoTokenizer

HUGGINGFACE_TOKENIZERS = {
    'chinese-character-bert': (BertTokenizer, "uer/gpt2-chinese-cluecorpussmall"),
    'xlm-250k': (AutoTokenizer, 'xlm-roberta-base')
}


def get_batch_function_for_multilingual_training(dataset, huge_pack_dir, data_bin_dtype, xlm_alpha, block_size, batch_size, device):
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    data_dir = os.path.join('data', dataset)
    langs = [n.split('_train.bin')[0] for n in os.listdir(data_dir) if n.endswith('_train.bin')]
    full_lang_dic = {
        lang: {
            'train': np.memmap(os.path.join(data_dir, f'{lang}_train.bin'), dtype=getattr(np, data_bin_dtype), mode='r'),
            'val': np.memmap(os.path.join(data_dir, f'{lang}_val.bin'), dtype=getattr(np, data_bin_dtype), mode='r')
        } for lang in langs
    }
    percentage_lang_dic = {}
    if huge_pack_dir and os.path.isdir(huge_pack_dir):
        percentage_lang_dic = {
            dir_: {
                'train': [
                    np.memmap(os.path.join(huge_pack_dir, dir_, f'{dir_}_{i}.bin'), dtype=getattr(np, data_bin_dtype), mode='r')
                    for i in range(99)
                ],
                'val': np.memmap(os.path.join(huge_pack_dir, dir_, f'{dir_}_99.bin'), dtype=getattr(np, data_bin_dtype), mode='r')
            } for dir_ in os.listdir(huge_pack_dir) if os.path.isdir(os.path.join(huge_pack_dir, dir_))
        }
    
    n_dict = {}
    for lang, v in full_lang_dic.items():
        n_dict[lang] = len(v['train'])
    for lang, v_dict in percentage_lang_dic.items():
        n_dict[lang] = sum(len(m) for m in v_dict['train'])
    p_dict = {lang: n / sum(n_dict.values()) for lang, n in n_dict.items()}
    q_dict = {lang: p ** xlm_alpha / sum(pp ** xlm_alpha for pp in p_dict.values()) for lang, p in p_dict.items()}

    def get_batch(split):
        lang = np.random.choice(list(q_dict.keys()), p=list(q_dict.values()))

        if lang in full_lang_dic:
            data = full_lang_dic[lang]['train'] if split == 'train' else full_lang_dic[lang]['val']
            max_pointer = len(data)
        else:
            data = percentage_lang_dic[lang]['train'][np.random.randint(len(percentage_lang_dic[lang]['train']))] \
                if split == 'train' else percentage_lang_dic[lang]['val']
            max_pointer = len(data) if split == 'train' else int(len(data) * 0.4)  # we only use first 40% on validation set
        ix = torch.randint(max_pointer - block_size, (batch_size,))
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
    # dataset, data_bin_dtype, xlm_alpha, block_size, batch_size, device
    get_batch = get_batch_function_for_multilingual_training('common_crawl', '/Users/hotaru/workspace/ai/cc100', 'uint32', 0.3, 1024, 2, 'cpu')
    for i in range(1000):
        print(get_batch('train'))
        print(get_batch('val'))
