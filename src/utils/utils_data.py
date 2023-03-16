import torch
from torch.utils.data import Dataset


class DLoader4Interpretation(Dataset):
    def __init__(self, data, tokenizer, config):
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.max_s = config.max_s
        self.max_len_per_utterance = int(self.max_len / self.max_s)
        self.data = [[self.tokenizer.encode(s)[:self.max_len_per_utterance] for s in d[:self.max_s]] for d in data]
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.length = len(self.data)

    def make_data(self, src, trg):
        # tokenizing src and trg
        src[0] = [self.cls_token_id] + src[0]
        src = [s + [self.sep_token_id] + [self.pad_token_id] * (self.max_len_per_utterance - 1 - len(s)) if len(
            s) < self.max_len_per_utterance else s[:self.max_len_per_utterance - 1] + [self.sep_token_id] for s in src]
        trg = trg + [self.sep_token_id] if len(trg) < self.max_len_per_utterance else trg[
                                                                                      :self.max_len_per_utterance - 1] + [
                                                                                          self.sep_token_id]

        # concatenating src and trg
        src = sum(src, [])
        pad_l = self.max_len - len(src) - len(trg)
        total = src + trg + [self.pad_token_id] * pad_l
        return total, len(src), len(src + trg)

    def __getitem__(self, idx):
        src, trg = self.data[idx][:-1], self.data[idx][-1]
        input, src_len, data_len = self.make_data(src, trg)
        return torch.LongTensor(input), src_len, data_len

    def __len__(self):
        return self.length


class DLoader4TESGAN(Dataset):
    def __init__(self, data, tokenizer, config):
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.data = sum(
            [[[self.cls_token_id] + self.tokenizer.encode(s)[:config.max_len - 2] + [self.sep_token_id] for s in d] for
             d in data], [])
        self.length = len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        s = s + [self.pad_token_id] * (self.max_len - len(s))
        return torch.LongTensor(s)

    def __len__(self):
        return self.length
