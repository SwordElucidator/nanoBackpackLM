import os
import json
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def cc_generator():
    path = 'data/common_crawl/'
    for f_name in os.listdir(os.path.join(path)):
        if f_name.endswith('.txt'):
            f_path = os.path.join(path, f_name)
            with open(f_path, 'r') as f:
                for line in f.readlines():
                    yield {'text':line.strip()}


dataset = Dataset.from_generator(cc_generator)
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
dataset = dataset.train_test_split(test_size=0.001, seed=2357, shuffle=True)  # keep for eval
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')


def process(example):
    ids = tokenizer.encode(example['text'])
    out = {'ids': ids, 'len': len(ids)}
    return out


# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    print(f"writing {filename}...")
    idx = 0
    for example in tqdm(dset):
        arr[idx: idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()
