import os
import sys
import random
import pickle
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append('src/')

import torch
import torch.nn as nn

from utils.utils_func import *
from bert_distances import FBD
from utils.config import Config
from tokenizer import Tokenizer
from multiset_distances import MultisetDistances
from models import InterpretationModel, Generator



def fbd(args):
    # path needed
    model = args.model
    testset_path = "./data/dailydialog/processed/dailydialog.test"
    data_path = {'test': testset_path}
    config_path = 'model/' + model + '/' + model + '.json'

    # init
    score_type = args.type
    n_samples = 1000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    try:
        config = Config(config_path)
    except:
        config = Config('src/config.json')
    tokenizer = Tokenizer(config).pretrained_tokenizer
    config.vocab_size = len(tokenizer)
    gpt2_path = find_detail_model(config.base_path, args.activation, config.gpt_model_size)

    # collect all sentences regardless dataset
    random.seed(999)
    dataset = {s: load_dataset(p) for s, p in data_path.items()}
    real = sum([[d for d in v] for v in dataset['test']], [])
    real = random.sample(real, n_samples)
    
    # init FBD
    scores = {}
    fbd = FBD(references=real, model_name="bert-base-uncased", bert_model_dir="/tmp/Bert/")

    torch.manual_seed(999)
    # load seed interpretation model
    gpt2 = InterpretationModel(config, device=device).to(device)
    check_point = torch.load(gpt2_path, map_location=device)
    gpt2.load_state_dict(check_point['model']['gpt2'])

    # make fixed noise and load the model
    np.random.seed(999)
    fixed_noise = torch.randn(n_samples, config.max_len, 768)
    fixed_noise = [fixed_noise[i * 100:(i + 1) * 100] for i in range(int(n_samples / 100))]
    
    # calculate FBD of random noise
    fake = sum([[tokenizer.decode(
        greedy_search(gpt2.model, tokenizer, n[j].unsqueeze(0).to(device), config.max_len,
                        config.activation, device, True)[0, config.max_len - 1:]) for j in range(n.size(0))] for n in
                fixed_noise], [])
    scores = fbd.get_score(sentences=fake)
    print(scores)

    # save the score
    save_path = 'score/' + 'randomNoise_' + score_type.upper() + '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(scores, f)



def msj(args):
    # path needed
    model = args.model
    testset_path = "./data/dailydialog/processed/dailydialog.test"
    data_path = {'test': testset_path}
    config_path = 'model/' + model + '/' + model + '.json'

    # init
    score_type = args.type
    n_samples = 1000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    try:
        config = Config(config_path)
    except:
        config = Config('src/config.json')
    tokenizer = Tokenizer(config).pretrained_tokenizer
    config.vocab_size = len(tokenizer)
    gpt2_path = find_detail_model(config.base_path, args.activation, config.gpt_model_size)

    # collect all sentences in test set
    random.seed(999)
    dataset = {s: load_dataset(p) for s, p in data_path.items()}
    real = sum([[d for d in v] for v in dataset['test']], [])
    real = random.sample(real, n_samples)
    real = [(''.join([c for c in s])).split() for s in real]

    # init MSJ
    scores = {}
    msd = MultisetDistances(references=real)

    torch.manual_seed(999)
    # load seed interpretation model
    gpt2 = InterpretationModel(config, device=device).to(device)
    check_point = torch.load(gpt2_path, map_location=device)
    gpt2.load_state_dict(check_point['model']['gpt2'])

    # make fixed noise and load the model
    np.random.seed(999)
    fixed_noise = torch.randn(n_samples, config.max_len, 768)
    fixed_noise = [fixed_noise[i * 100:(i + 1) * 100] for i in range(int(n_samples / 100))]

    # calculate MSJ of random noise
    fake = sum([[tokenizer.decode(
        greedy_search(gpt2.model, tokenizer, n[j].unsqueeze(0).to(device), config.max_len,
                        config.activation, device, True)[0, config.max_len - 1:]) for j in range(n.size(0))] for n in
                fixed_noise], [])
    fake = [(''.join([c for c in s])).split() for s in fake]
    scores = msd.get_jaccard_score(sentences=fake)
    print(scores)

    # save the score
    save_path = 'score/' + 'randomNoise_' + score_type.upper() + '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(scores, f)



def dsr(args):
    # path needed
    model = args.model
    trainset_path = "./data/dailydialog/processed/dailydialog.train"
    data_path = {'train': trainset_path}
    config_path = 'model/' + model + '/' + model + '.json'

    # init
    repeat_rate = {}
    score_type = args.type
    n_samples = 1000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    try:
        config = Config(config_path)
    except:
        config = Config('src/config.json')
    tokenizer = Tokenizer(config).pretrained_tokenizer
    config.vocab_size = len(tokenizer)
    gpt2_path = find_detail_model(config.base_path, args.activation, config.gpt_model_size)

    # collect all sentences in training set
    random.seed(999)
    dataset = {s: load_dataset(p) for s, p in data_path.items()}
    real = sum([[d for d in v] for v in dataset['train']], [])
    real = [' '.join([str(i) for i in tokenizer.encode(s)]) for s in real]

    torch.manual_seed(999)
    # load seed interpretation model
    gpt2 = InterpretationModel(config, device=device).to(device)
    check_point = torch.load(gpt2_path, map_location=device)
    gpt2.load_state_dict(check_point['model']['gpt2'])

    # make fixed noise and load the model
    np.random.seed(999)
    fixed_noise = torch.randn(n_samples, config.max_len, 768)
    fixed_noise = [fixed_noise[i * 100:(i + 1) * 100] for i in range(int(n_samples / 100))]

    # check repeat rate of random noise
    repeat_num = 0
    fake = sum([[greedy_search(gpt2.model, tokenizer, n[j].unsqueeze(0).to(device), config.max_len,
                                config.activation, device, True)[0, config.max_len - 1:] for j in range(n.size(0))] for n
                in fixed_noise], [])
    fake = [' '.join([str(i) for i in d.cpu().tolist()]) for d in fake]

    for s in fake:
        if s in real and len(s.split()) > 10:
            repeat_num += 1

    dup, div = 1 - repeat_num / len(fake), len(set(fake)) / len(fake)
    repeat_rate = (dup, div, dup * div, 2 * dup * div / (dup + div))
    print(repeat_rate)

    # save the repeat rate
    save_path = 'score/' + 'randomNoise_' + score_type.upper() + '.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(repeat_rate, f)




def lm(args):
    # path needed
    model = args.model

    # init
    score_type = args.type
    n_samples = 1000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # define model for LM score
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
    lm_model.eval()
    criterion = nn.CrossEntropyLoss()

    # define required
    random.seed(999)
    total_loss, total_num = 0, 0

    ########################### case of ours ###########################
    if model not in ['seqgan', 'rankgan', 'maligan', 'mle', 'pgbleu']:
        # load synthesized samples
        path = './syn/' + model + '.txt'
        with open(path, 'r') as f:
            data = f.readlines()
        
        total_loss = 0
        total_num = 0
        data = [d.strip() for d in data]
        for s in tqdm(data):
            try:
                s = torch.LongTensor(tokenizer.encode(s)).unsqueeze(0)
                output = lm_model(s).logits
                loss = criterion(output[:, :-1, :].reshape(-1, output.size(-1)), s[:, 1:].reshape(-1))
                if not np.isnan(loss.item()):
                    total_loss += loss.item()
                    total_num += 1
            except:
                continue
        lm_score = total_loss/total_num
        print(lm_score)

        # save the data
        save_path = 'score/' + model + '_' + score_type.upper() + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(lm_score, f)


    ########################### case of others ###########################
    elif model in ['seqgan', 'rankgan', 'maligan', 'mle', 'pgbleu']:
        names = {'seqgan': 'SeqGAN', 'rankgan': 'RankGAN', 'maligan': 'MaliGAN', 'mle': 'MLE', 'pgbleu': 'PG_Bleu'}
        
        # load synthesized samples
        path = './model/' + model + '/generator_sample0.txt'
        with open(path, 'r') as f:
            data = f.readlines()
        data = random.sample(data, 1000)
        data = [d.strip() for d in data]

        # calculate lm score
        for tok in tqdm(data):
            tok = list(filter(lambda x: x != 50259, map(int, tok.split())))
            s = tokenizer.decode(tok)
            s = torch.LongTensor(tokenizer.encode(s)).unsqueeze(0)
            output = lm_model(s).logits
            loss = criterion(output[:, :-1, :].reshape(-1, output.size(-1)), s[:, 1:].reshape(-1))
            if not np.isnan(loss.item()):
                total_loss += loss.item()
                total_num += 1
        
        lm_score = total_loss / total_num
        print(lm_score)
        
        # save the data
        save_path = 'score/' + names[model] + '_' + score_type.upper() + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(lm_score, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-t', '--type', type=str, required=True)
    parser.add_argument('-a', '--activation', default='sigmoid', type=str, required=False,
                        choices=['sigmoid', 'tanh', 'none'])
    args = parser.parse_args()

    assert os.path.isdir('model/' + args.model)
    if args.type.lower() == 'fbd':
        fbd(args)
    elif args.type.lower() == 'msj':
        msj(args)
    elif args.type.lower() == 'dsr':
        dsr(args)
    elif args.type.lower() == 'lm':
        lm(args)
    else:
        print('FBD, MSJ, DSR, LM are possible, please check...')
        assert AssertionError
