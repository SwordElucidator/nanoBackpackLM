import os
import json
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from functools import partial


langs = [
    'af',
 'am',
 'ar',
 'as',
 'az',
 'be',
 'bg',
 'bn',
 'bn_rom',
 'br',
 'bs',
 'ca',
 'cs',
 'cy',
 'da',
 'de',
 'el',
 'en',
 'eo',
 'es',
 'et',
 'eu',
 'fa',
 'ff',
 'fi',
 'fr',
 'fy',
 'ga',
 'gd',
 'gl',
 'gn',
 'gu',
 'ha',
 'he',
 'hi',
 'hi_rom',
 'hr',
 'ht',
 'hu',
 'hy',
 'id',
 'ig',
 'is',
 'it',
 'ja',
 'jv',
 'ka',
 'kk',
 'km',
 'kn',
 'ko',
 'ku',
 'ky',
 'la',
 'lg',
 'li',
 'ln',
 'lo',
 'lt',
 'lv',
 'mg',
 'mk',
 'ml',
 'mn',
 'mr',
 'ms',
 'my',
 'my_zaw',
 'ne',
 'nl',
 'no',
 'ns',
 'om',
 'or',
 'pa',
 'pl',
 'ps',
 'pt',
 'qu',
 'rm',
 'ro',
 'ru',
 'sa',
 'si',
 'sc',
 'sd',
 'sk',
 'sl',
 'so',
 'sq',
 'sr',
 'ss',
 'su',
 'sv',
 'sw',
 'ta',
 'ta_rom',
 'te',
 'te_rom',
 'th',
 'tl',
 'tn',
 'tr',
 'ug',
 'uk',
 'ur',
 'ur_rom',
 'uz',
 'vi',
 'wo',
 'xh',
 'yi',
 'yo',
 'zh-Hans',
 'zh-Hant',
 'zu']

langs = langs[0: 30]
num_proc = 8

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')


def process(example):
    ids = tokenizer.encode(example['text'])
    return {'ids': ids, 'len': len(ids)}


path = 'data/common_crawl/'
os.listdir(path)
existed = [i.split('_train')[0] for i in os.listdir(path) if '_train.bin' in i]


for lang in langs:
    if lang in existed:
        continue
    dataset = load_dataset("cc100", lang=lang)['train']
    _eval_size = 10000 if dataset.num_rows * 0.001 < 10000 else 0.001
    _val_size = 5000 if dataset.num_rows * 0.0005 < 5000 else 0.0005
    dataset = dataset.train_test_split(test_size=_eval_size + _val_size, seed=2357)
    train_set = dataset['train']
    split_dataset = dataset["test"].train_test_split(test_size=_val_size / (_val_size + _eval_size), seed=2357,
                                                     shuffle=False)
    split_dataset['val'] = split_dataset.pop('test')
    split_dataset['evaluation'] = split_dataset.pop('train')
    split_dataset['train'] = train_set
    tokenized = split_dataset.map(
        process,
        remove_columns=['text', 'id'],
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
