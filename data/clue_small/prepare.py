import os
import json
import pdb

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

num_proc = 8


# for chinese, we will use huggingface tokenizer


def wiki_generator():
    path = 'data/clue_small/wiki_zh'
    for subdir in os.listdir(os.path.join(path)):
        subdir = os.path.join(path, subdir)
        if os.path.isdir(subdir):
            for f_name in os.listdir(subdir):
                if f_name.startswith('wiki'):
                    with open(os.path.join(subdir, f_name), 'r') as f:
                        for line in f.readlines():
                            yield {'text': json.loads(line)['text'].strip().split('\n\n', 2)[-1]}


dataset = Dataset.from_generator(wiki_generator)
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

dataset = dataset.train_test_split(test_size=0.001, seed=2357, shuffle=True)
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')  # rename the test split to val


def process(example):
    ids = tokenizer.encode(example['text'])
    ids.append(tokenizer.sep_token_id)
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
