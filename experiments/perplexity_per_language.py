import os

import numpy as np
import torch

from experiments.loader import eval_iters, evaluation_data_dirs, load_model, get_batch, device_type, dtype, \
    data_bin_dtype
from contextlib import nullcontext


ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def percentage_data_sample_func(data, block_size, batch_size):
    return torch.randint(int(len(data) * 0.6) - block_size, (batch_size,)) + int(len(data) * 0.4)


def evaluate_single_language(evaluation_data_path, model, is_percentage):
    eval_data = np.memmap(evaluation_data_path, dtype=getattr(np, data_bin_dtype), mode='r')
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(eval_data, sample_func=percentage_data_sample_func if is_percentage else None)
        with ctx:
            _, loss = model(X, Y)
        losses[k] = torch.exp(loss).item()
    return losses.mean()


@torch.no_grad()
def estimate_perplexity():
    model = load_model()[0]
    evaluation_output = {}
    for dir_ in evaluation_data_dirs.split(','):
        for f_name in os.listdir(dir_):
            if f_name.endswith('_evaluation.bin'):
                is_percentage = False
                lang_name = f_name.split('_evaluation')[0]
            elif os.path.isdir(os.path.join(dir_, f_name)):
                is_percentage = True
                lang_name = f_name
                f_name = os.path.join(f_name, f'{f_name}_99.bin')
            else:
                continue
            perp = evaluate_single_language(os.path.join(dir_, f_name), model, is_percentage)
            print('perplexity for {} is {}'.format(lang_name, perp))
            evaluation_output[lang_name] = perp
    return evaluation_output


if __name__ == "__main__":
    print(estimate_perplexity())
