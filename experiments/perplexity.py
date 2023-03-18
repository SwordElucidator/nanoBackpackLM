import numpy as np
import torch
from transformers import GPT2LMHeadModel

from experiments.loader import eval_iters, evaluation_data_path, load_model, get_batch, device_type, dtype
from contextlib import nullcontext


ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


@torch.no_grad()
def estimate_perplexity(from_huggingface_model_name=None):
    if from_huggingface_model_name:
        model = GPT2LMHeadModel.from_pretrained(from_huggingface_model_name).to(device_type)
    else:
        model = load_model()[0]
    eval_data = np.memmap(evaluation_data_path, dtype=np.uint16, mode='r')
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(eval_data)
        with ctx:
            if from_huggingface_model_name:
                loss = model(input_ids=X, labels=Y).loss
            else:
                _, loss = model(X, Y)
        losses[k] = torch.exp(loss).item()
    return losses.mean()


if __name__ == "__main__":
    # print(f'perplexity is {estimate_perplexity()}')
    print(f'perplexity is {estimate_perplexity("uer/gpt2-chinese-cluecorpussmall")}')
