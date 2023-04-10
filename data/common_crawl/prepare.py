import os
import json
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from functools import partial


num_proc = 8


langs = ['af', 'am']


def cc_language_based_generator(lang):
    path = 'data/common_crawl/'
    f_path = os.path.join(path, f"{lang}.txt")
    with open(f_path, 'r') as f:
        for line in f.readlines():
            yield {'text': line.strip()}


def process(example):
    ids = tokenizer.encode(example['text'])
    out = {'ids': ids, 'len': len(ids)}
    return out


tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
for lang in langs:
    dataset = Dataset.from_generator(partial(cc_language_based_generator, lang))
    dataset = dataset.train_test_split(test_size=0.001, seed=2357, shuffle=True)  # keep for eval
    eval_set = dataset['test']
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
    split_dataset['evaluation'] = eval_set

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
