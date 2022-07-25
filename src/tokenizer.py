from transformers import GPT2Tokenizer


class Tokenizer:
    def __init__(self, config):
        self.pretrained_tokenizer = GPT2Tokenizer.from_pretrained(config.gpt_model_size)
        self.pretrained_tokenizer.add_special_tokens({'cls_token': '<cls>', 'sep_token': '<sep>', 'pad_token': '<pad>'})
