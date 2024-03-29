import os
import pickle

import numpy as np
import torch
import tiktoken
from backpack import BackpackLM, BackpackLMConfig
from model import GPT, GPTConfig
from task_utils import HUGGINGFACE_TOKENIZERS

# -----------------------------------------------------------------------------
out_dir = 'out'  # ignored if init_from is not 'resume'
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
compile = False  # use PyTorch 2.0 to compile the model to be faster
tokenizer_name = 'gpt2'
dtype = 'bfloat16'
bias = False
strict = True
data_bin_dtype = 'uint16'


# other params
evaluation_data_path = 'data/clue_small/evaluation.bin'
evaluation_data_dirs = 'data/common_crawl'
eval_iters = 500
block_size = 1024
batch_size = 12
model_name = 'backpack-lm'

exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
Model = BackpackLM if model_name == 'backpack-lm' else GPT
Config = BackpackLMConfig if model_name == 'backpack-lm' else GPTConfig

def load_model(force_model=None, force_dir=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(force_dir if force_dir else out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    conf = Config(**checkpoint['model_args'])
    model = (force_model if force_model else Model)(conf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:  # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    elif tokenizer_name in HUGGINGFACE_TOKENIZERS.keys():
        print(f"Use {tokenizer_name} encodings...")
        klass, h_tokenizer_name = HUGGINGFACE_TOKENIZERS[tokenizer_name]
        tokenizer = klass.from_pretrained(h_tokenizer_name)
        encode = lambda s: tokenizer.encode(s)
        decode = lambda l: tokenizer.decode(l)
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    return model, encode, decode


def get_batch(data, sample_func=None):
    if sample_func:
        ix = sample_func(data, block_size, batch_size)
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
