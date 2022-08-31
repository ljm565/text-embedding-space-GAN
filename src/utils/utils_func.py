import torch
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.nist_score import corpus_nist
import pickle
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F


def save_checkpoint(file, models, optimizers):
    state = {'model': models, 'optimizer': optimizers}
    torch.save(state, file)


def bleu_score(ref, ans, weights=(0.25, 0.25, 0.25, 0.25)):
    return corpus_bleu(ref, ans, weights)


def nist_score(ref, ans, n):
    return corpus_nist(ref, ans, n)


def cal_ppl(loss):
    return np.exp(loss)


def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_loss_history(target, losses, dataset_len):
    for k, v in losses.items():
        target[k].append(sum(v) / dataset_len)
    return target


def save_val_s(b_size, val_s, output, input, src_len, data_len, tokenizer):
    output, input, src_len = output.detach().cpu(), input.detach().cpu(), src_len.detach().cpu()
    for i in range(b_size):
        st, tr = src_len[i], data_len[i] - 1
        trg = tokenizer.decode(input[i, st:tr].tolist())
        out = tokenizer.decode(torch.argmax(output[i, st - 1:tr - 1], dim=1).tolist())
        val_s.append((trg, out))
    return val_s


def save_score_history(score_history, val_s):
    ref, hypo = [], []
    for trg, out in val_s:
        ref.append([trg.split()])
        hypo.append(out.split())

    score_history['bleu-2'].append(bleu_score(ref, hypo, weights=(0.5, 0.5)))
    score_history['bleu-4'].append(bleu_score(ref, hypo, weights=(0.25, 0.25, 0.25, 0.25)))
    score_history['nist-2'].append(nist_score(ref, hypo, 2))
    score_history['nist-4'].append(nist_score(ref, hypo, 4))
    return score_history


def logger(k, v):
    print('{}: {:.5f}'.format(k, v))


def print_model_save(score_type):
    print('{} best model pt file is being saved'.format(score_type))


def save_model(model_path, score_curr, score_best, score_type, model, optimizer, epoch):
    if score_type == 'ppl':
        if score_curr < score_best:
            score_best = score_curr
            saving(model_path, score_type, model, optimizer, epoch)
    else:
        if score_curr > score_best:
            score_best = score_curr
            saving(model_path, score_type, model, optimizer, epoch)
    return score_best


def saving(model_path, score_type, model, optimizer, epoch):
    try:
        old_model = list(
            filter(lambda x: x.endswith(score_type + '.pt'), os.listdir(model_path[:model_path.rfind('/') + 1] + '/')))[0]
        old_model = model_path[:model_path.rfind('/') + 1] + '/' + old_model
        os.remove(old_model)
    except IndexError:
        pass

    model_path = model_path + '_' + str(epoch + 1) + '_' + score_type + '.pt'
    model_wts = {'gpt2': model.state_dict()}
    optimizer_wts = {'gpt2': optimizer.state_dict()}
    print_model_save(score_type)
    save_checkpoint(model_path, model_wts, optimizer_wts)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.08)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.08)


def greedy_search(model, tokenizer, input, max_len, act, device, is_emb=False):
    emb_layer = model.resize_token_embeddings()
    hiddens = model.transformer.h
    tanh = nn.Tanh()
    sigmoid = nn.Sigmoid()

    if is_emb:
        for _ in range(max_len):
            for i, l in enumerate(hiddens):
                vocab_output = l(input)[0] if i == 0 else l(vocab_output)[0]
            vocab_output = model.transformer.ln_f(vocab_output)
            vocab_output = model.lm_head(vocab_output)
            last_id = torch.argmax(vocab_output, dim=-1)[:, -1].unsqueeze(0)
            output_id = emb_layer(last_id) + model.transformer.wpe(
                torch.LongTensor([input.size(1)]).to(device)).unsqueeze(0)
            input = torch.cat((input, output_id), dim=1)
            if last_id == tokenizer.sep_token_id:
                break

        for l in hiddens:
            input = l(input)[0]
        input = model.transformer.ln_f(input)
        input = model.lm_head(input)
        input = torch.argmax(input, dim=-1)
        input = input[:, :-2]
    else:
        for _ in range(max_len):
            if act == 'sigmoid':
                tmp = sigmoid(model.transformer.wte(input) + model.transformer.wpe(
                    torch.arange(input.size(1)).to(device)).unsqueeze(0))
            elif act == 'tanh':
                tmp = tanh(model.transformer.wte(input) + model.transformer.wpe(
                    torch.arange(input.size(1)).to(device)).unsqueeze(0))
            elif act == 'none' or act == 'None':
                tmp = model.transformer.wte(input) + model.transformer.wpe(
                    torch.arange(input.size(1)).to(device)).unsqueeze(0)

            for l in hiddens:
                tmp = l(tmp)[0]
            output_id = model.transformer.ln_f(tmp)
            output_id = model.lm_head(output_id)
            output_id = torch.argmax(output_id, dim=-1)[:, -1].unsqueeze(0)

            input = torch.cat((input, output_id), dim=1)
            if output_id == tokenizer.sep_token_id:
                break
        input = input[:, :-1]
    return input


def prob_search(model, tokenizer, input, max_len, act, device, is_emb=False):
    emb_layer = model.resize_token_embeddings()
    hiddens = model.transformer.h
    tanh = nn.Tanh()
    sigmoid = nn.Sigmoid()

    if is_emb:
        for st in range(max_len):
            for i, l in enumerate(hiddens):
                vocab_output = l(input)[0] if i == 0 else l(vocab_output)[0]
            vocab_output = model.transformer.ln_f(vocab_output)
            vocab_output = model.lm_head(vocab_output)
            last_id = torch.multinomial(F.softmax(vocab_output[:, -1], dim=-1), 1) if st == 0 else torch.argmax(
                vocab_output, dim=-1)[:, -1].unsqueeze(0)
            output_id = emb_layer(last_id) + model.transformer.wpe(
                torch.LongTensor([input.size(1)]).to(device)).unsqueeze(0)
            input = torch.cat((input, output_id), dim=1)
            if last_id == tokenizer.sep_token_id:
                break

        for l in hiddens:
            input = l(input)[0]
        input = model.transformer.ln_f(input)
        input = model.lm_head(input)
        input = torch.argmax(input, dim=-1)
        input = input[:, :-2]
    else:
        for st in range(max_len):
            if act == 'sigmoid':
                tmp = sigmoid(model.transformer.wte(input) + model.transformer.wpe(
                    torch.arange(input.size(1)).to(device)).unsqueeze(0))
            elif act == 'tanh':
                tmp = tanh(model.transformer.wte(input) + model.transformer.wpe(
                    torch.arange(input.size(1)).to(device)).unsqueeze(0))
            elif act == 'none' or act == 'None':
                tmp = model.transformer.wte(input) + model.transformer.wpe(
                    torch.arange(input.size(1)).to(device)).unsqueeze(0)

            for l in hiddens:
                tmp = l(tmp)[0]
            output_id = model.transformer.ln_f(tmp)
            output_id = model.lm_head(output_id)
            output_id = torch.multinomial(F.softmax(output_id[:, -1], dim=-1), 1) if st == 0 else torch.argmax(
                output_id, dim=-1)[:, -1].unsqueeze(0)

            input = torch.cat((input, output_id), dim=1)
            if output_id == tokenizer.sep_token_id:
                break
        input = input[:, :-1]
    return input


def make_noise(b_size, init_size):
    t = [torch.Tensor(np.random.uniform(-10, 10, init_size)) for _ in range(b_size)]
    return torch.cat(tuple(t), dim=0).view(b_size, init_size, 1, 1)


def find_detail_model(base_path, activation, score='bleu4'):
    assert score in ['ppl', 'bleu2', 'bleu4', 'nist2', 'nist4', 'last']
    activation = activation.lower()
    base_path = base_path + 'model/' + 'interp_' + activation + '/'
    detail_path = base_path + list(filter(lambda x: x.endswith(score + '.pt'), os.listdir(base_path)))[0]
    return detail_path
