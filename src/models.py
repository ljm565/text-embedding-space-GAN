from transformers import GPT2LMHeadModel, BertForNextSentencePrediction, BertConfig
import torch
import torch.nn as nn
from utils.utils_func import *


class InterpretationModel(nn.Module):
    def __init__(self, config, pad_idx, device):
        super(InterpretationModel, self).__init__()
        self.pad_idx = pad_idx
        self.device = device
        self.gpt_model_size = config.gpt_model_size
        self.model = GPT2LMHeadModel.from_pretrained(self.gpt_model_size, output_hidden_states=True)
        self.embedding_layer = self.model.resize_token_embeddings(config.vocab_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.max_len = config.max_len
        self.activation = config.activation

    def forward(self, input, emb=False):
        real_data = None

        if emb == False:
            w_emb = self.model.transformer.wte(input)
            p_emb = self.model.transformer.wpe(torch.arange(self.max_len).to(self.device)).unsqueeze(0)

            if self.activation == 'sigmoid':
                real_data = self.sigmoid(w_emb + p_emb)
            elif self.activation == 'tanh':
                real_data = self.tanh(w_emb + p_emb)
            elif self.activation == 'none' or self.activation == 'None':
                real_data = w_emb + p_emb

        for i, l in enumerate(self.model.transformer.h):
            vocab_output = l(input if emb else real_data)[0] if i == 0 else l(vocab_output)[0]
        vocab_output = self.model.transformer.ln_f(vocab_output)
        vocab_output = self.model.lm_head(vocab_output)

        return real_data, vocab_output


class Generator(nn.Module):
    def __init__(self, config, device):
        super(Generator, self).__init__()
        self.noise_init_size = config.noise_init_size
        self.max_len = config.max_len
        self.activation = config.activation
        self.perturbed = config.perturbed
        self.device = device
        self.layers = [
            nn.ConvTranspose1d(in_channels=self.noise_init_size, out_channels=128, kernel_size=256, stride=1,
                               padding=0, bias=True),
            nn.LeakyReLU(0.5),
            nn.ConvTranspose1d(in_channels=128, out_channels=self.max_len, kernel_size=3, stride=3, padding=0,
                               bias=True)
        ]
        if self.activation == 'sigmoid':
            self.layers.append(nn.Sigmoid())
        elif self.activation == 'tanh':
            self.layers.append(nn.Tanh())

        self.generator = nn.Sequential(*self.layers)

    def forward(self, x):
        b_size = x.size(0)
        x = self.generator(x)

        if self.perturbed:
            z = torch.randn(b_size, self.max_len, 768).to(self.device)
            x = x + z
        return x


class Discriminator_BERT(nn.Module):
    def __init__(self, config):
        super(Discriminator_BERT, self).__init__()
        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            num_hidden_layers=config.bert_layer)
        self.model = BertForNextSentencePrediction(bert_config)

    def forward(self, input, label):
        output = self.model(
            inputs_embeds=input,
            labels=label)
        return output.loss, output.logits


class Discriminator_LSTM(nn.Module):
    def __init__(self, device):
        super(Discriminator_LSTM, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(768 * 4, 1),
            nn.Sigmoid()
        )

    def init_hidden(self, b_size):
        h0 = torch.zeros(2 * 2, b_size, 768).to(self.device)
        c0 = torch.zeros(2 * 2, b_size, 768).to(self.device)
        return h0, c0

    def forward(self, input):
        b_size = input.size(0)
        h0, c0 = self.init_hidden(b_size)
        lstm_out, _ = self.lstm(input, (h0, c0))
        lstm_out = torch.cat((lstm_out[:, 0, :], lstm_out[:, -1, :]), dim=-1)
        lstm_out = self.fc(lstm_out).squeeze()
        return lstm_out
