from transformers import BertTokenizer, AutoTokenizer

HUGGINGFACE_TOKENIZERS = {
    'chinese-character-bert': (BertTokenizer, "uer/gpt2-chinese-cluecorpussmall"),
    'xlm-250k': (AutoTokenizer, 'xlm-roberta-base')
}