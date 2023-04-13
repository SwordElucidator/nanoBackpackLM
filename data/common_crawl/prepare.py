import os
import json
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from functools import partial


num_proc = 40


path = 'data/common_crawl/'


def cc_language_based_generator(f):
    for line in f.readlines():
        yield {'text': line.strip()}


def process(example):
    ids = tokenizer.encode(example['text'])
    out = {'ids': ids, 'len': len(ids)}
    return out


tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

for f_name in os.listdir(path):
    if f_name.endswith('txt.xz'):
        lang = f_name.replace('.txt.xz', '')
        zip_path = os.path.join(path, f"{lang}.txt.xz")
        os.system(f"xz -d {zip_path}")  # unzip
    elif f_name.endswith('.txt'):
        lang = f_name.replace('.txt', '')
    else:
        continue
    f_path = os.path.join(path, f"{lang}.txt")
    with open(f_path, 'r') as f:
        dataset = Dataset.from_generator(partial(cc_language_based_generator, f))
    _eval_size = 10000 if dataset.num_rows * 0.001 < 10000 else 0.001
    _val_size = 5000 if dataset.num_rows * 0.0005 < 5000 else 0.0005
    dataset = dataset.train_test_split(test_size=_eval_size + _val_size, seed=2357, shuffle=dataset.num_rows < 10000000)  # we assume that very large corpus is already "shuffled"
    train_set = dataset['train']
    split_dataset = dataset["test"].train_test_split(test_size=_val_size / (_val_size + _eval_size), seed=2357, shuffle=False)
    split_dataset['val'] = split_dataset.pop('test')
    split_dataset['evaluation'] = split_dataset.pop('train')
    split_dataset['train'] = train_set

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'])
        filename = os.path.join(os.path.dirname(__file__), f'{lang}_{split}.bin')
        dtype = np.uint32  # well, we have so much tokens...
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        print(f"writing {filename}...")
        idx = 0
        for example in tqdm(dset):
            arr[idx: idx + example['len']] = example['ids']
            idx += example['len']
        arr.flush()
    os.remove(f_path)  # delete txt
