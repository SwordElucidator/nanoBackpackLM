# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'backpack-xlm-micro'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5
n_layer = 6
n_head = 6
n_embd = 384

# this makes total number of tokens be 300B
max_iters = 200000
lr_decay_iters = 200000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

out_dir = 'backpack_xlm'
dataset = 'common_crawl'
tokenizer_name = 'xlm-250k'
huge_pack_dir = '/home/ubuntu/workspace/ai/cc100'  # FIX WITH YOUR PATH
data_bin_dtype = 'uint32'
xlm_alpha = 0.3
