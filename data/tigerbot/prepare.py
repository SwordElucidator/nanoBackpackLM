# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
from transformers import BertTokenizer

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
dataset = load_dataset("TigerResearch/pretrain_zh")['train']
dataset = dataset.train_test_split(test_size=0.001, seed=2357, shuffle=True)
split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=False)
split_dataset['val'] = split_dataset.pop('test') # rename the test split to val


tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

def process(example):
    ids = tokenizer.encode(example['content'])
    out = {'ids': ids, 'len': len(ids)}
    return out


# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['dataType', 'title', 'content', 'uniqueKey', 'titleUkey', 'id'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    print(f"writing {filename}...")
    idx = 0
    for example in tqdm(dset):
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()

# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')
