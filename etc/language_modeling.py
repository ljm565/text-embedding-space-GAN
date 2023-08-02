import torch
import torch.nn as nn

import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer



class LanguageModeling:
    def __init__(self, device):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.lm_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.lm_model.eval()
        self.criterion = nn.CrossEntropyLoss()

    
    def get_lm(self, sentences):
        total_loss, total_num = 0, 0
        sentences = [d.strip() for d in sentences]
        for s in sentences:
            try:
                s = torch.LongTensor(self.tokenizer.encode(s)).unsqueeze(0).to(self.device)
                output = self.lm_model(s).logits
                loss = self.criterion(output[:, :-1, :].reshape(-1, output.size(-1)), s[:, 1:].reshape(-1))
                if not np.isnan(loss.item()):
                    total_loss += loss.item()
                    total_num += 1
            except:
                continue
        lm_score = total_loss/total_num

        return lm_score