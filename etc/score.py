import sys
import os
from tqdm import tqdm
import random
import pickle
from argparse import ArgumentParser

sys.path.append('src/')

import torch
import numpy as np
from utils.config import Config
from models import InterpretationModel, Generator
from tokenizer import Tokenizer
from utils.utils_func import *
from bert_distances import FBD
from multiset_distances import MultisetDistances


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
    gpt2_path = find_detail_model(config.base_path, args.activation)

    # collect all sentences regardless dataset
    random.seed(999)
    dataset = {s: load_dataset(p) for s, p in data_path.items()}
    real = sum([[d for d in v] for v in dataset['test']], [])
    real = random.sample(real, n_samples)

    # init FBD
    scores = {}
    fbd = FBD(references=real, model_name="bert-base-uncased", bert_model_dir="/tmp/Bert/")

    ########################### case of ours ###########################
    if model not in ['seqgan', 'rankgan', 'maligan', 'mle', 'pgbleu']:
        torch.manual_seed(999)
        # load seed interpretation model
        gpt2 = InterpretationModel(config, pad_idx=tokenizer.pad_token_id, device=device).to(device)
        check_point = torch.load(gpt2_path, map_location=device)
        gpt2.load_state_dict(check_point['model']['gpt2'])

        # make fixed noise and load the model
        np.random.seed(999)
        fixed_noise = make_noise(n_samples, config.noise_init_size)
        fixed_noise = [fixed_noise[i * 100:(i + 1) * 100] for i in range(int(n_samples / 100))]
        generator = Generator(config).to(device)
        base_path = './model/' + model
        model_path = [base_path + '/' + model + '_' + str(i + 1) + '.pt' for i in range(len(os.listdir(base_path)) - 1)]

        # calculate FBD
        for p in tqdm(model_path):
            check_point = torch.load(p, map_location=device)
            generator.load_state_dict(check_point['model']['generator'])
            generator.eval()
            fake = sum([[tokenizer.decode(
                greedy_search(gpt2.model, tokenizer, generator(n.to(device))[j].unsqueeze(0), config.max_len,
                              config.activation, device, True)[0, config.max_len - 1:]) for j in range(n.size(0))] for n in
                        fixed_noise], [])
            scores[p] = fbd.get_score(sentences=fake)
            print(scores[p])

        # save the score
        save_path = 'score/' + model + '_' + score_type.upper() + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(scores, f)

    ########################### case of others ###########################
    elif model in ['seqgan', 'rankgan', 'maligan', 'mle', 'pgbleu']:
        names = {'seqgan': 'SeqGAN', 'rankgan': 'RankGAN', 'maligan': 'MaliGAN', 'mle': 'MLE', 'pgbleu': 'PG_Bleu'}
        model_path = ['model/' + model + '/generator_sample' + str(i) + '.txt' for i in
                      range(len(list(filter(lambda x: x.startswith('generator'), os.listdir('model/' + model)))))]
        for path in model_path:
            fake = []
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    line = line.split(' ')
                    line = [int(c) for c in line]
                    for j in range(len(line)):
                        if line[j] == tokenizer.pad_token_id:
                            line = line[:j]
                            break
                    fake.append(tokenizer.decode(line[:j]))
            fake = random.sample(fake, n_samples)
            scores[path] = fbd.get_score(sentences=fake)
            print(scores[path])

        # save the score 
        save_path = 'score/' + names[model] + '_' + score_type.upper() + '.pkl'
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
    gpt2_path = find_detail_model(config.base_path, args.activation)

    # collect all sentences in test set
    random.seed(999)
    dataset = {s: load_dataset(p) for s, p in data_path.items()}
    real = sum([[d for d in v] for v in dataset['test']], [])
    real = random.sample(real, n_samples)
    real = [(''.join([c for c in s])).split() for s in real]

    # init MSJ
    scores = {}
    msd = MultisetDistances(references=real)

    ########################### case of ours ###########################
    if model not in ['seqgan', 'rankgan', 'maligan', 'mle', 'pgbleu']:
        torch.manual_seed(999)
        # load seed interpretation model
        gpt2 = InterpretationModel(config, pad_idx=tokenizer.pad_token_id, device=device).to(device)
        check_point = torch.load(gpt2_path, map_location=device)
        gpt2.load_state_dict(check_point['model']['gpt2'])

        # make fixed noise and load the model
        np.random.seed(999)
        fixed_noise = make_noise(n_samples, config.noise_init_size)
        fixed_noise = [fixed_noise[i * 100:(i + 1) * 100] for i in range(int(n_samples / 100))]
        generator = Generator(config).to(device)
        base_path = './model/' + model
        model_path = [base_path + '/' + model + '_' + str(i + 1) + '.pt' for i in range(len(os.listdir(base_path)) - 1)]

        # calculate MSJ
        for p in tqdm(model_path):
            check_point = torch.load(p, map_location=device)
            generator.load_state_dict(check_point['model']['generator'])
            generator.eval()
            fake = sum([[tokenizer.decode(
                greedy_search(gpt2.model, tokenizer, generator(n.to(device))[j].unsqueeze(0), config.max_len,
                              config.activation, device, True)[0, config.max_len - 1:]) for j in range(n.size(0))] for n in
                        fixed_noise], [])
            fake = [(''.join([c for c in s])).split() for s in fake]
            scores[p] = msd.get_jaccard_score(sentences=fake)
            print(scores[p])

        # save the score
        save_path = 'score/' + model + '_' + score_type.upper() + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(scores, f)

    ########################### case of others ###########################
    elif model in ['seqgan', 'rankgan', 'maligan', 'mle', 'pgbleu']:
        names = {'seqgan': 'SeqGAN', 'rankgan': 'RankGAN', 'maligan': 'MaliGAN', 'mle': 'MLE', 'pgbleu': 'PG_Bleu'}
        train = []
        with open('model/' + model + '/real_train_dailydialog.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                train.append(line)

        model_path = ['model/' + model + '/generator_sample' + str(i) + '.txt' for i in
                      range(len(list(filter(lambda x: x.startswith('generator'), os.listdir('model/' + model)))))]
        for path in model_path:
            fake = []
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    line = line.split(' ')
                    line = [int(c) for c in line]
                    for j in range(len(line)):
                        if line[j] == tokenizer.pad_token_id:
                            line = line[:j]
                            break
                    fake.append(tokenizer.decode(line[:j]))
            fake = random.sample(fake, n_samples)
            fake = [(''.join([c for c in s])).split() for s in fake]
            scores[path] = msd.get_jaccard_score(sentences=fake)
            print(scores[path])

        # save the score 
        save_path = 'score/' + names[model] + '_' + score_type.upper() + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(scores, f)


def sdr(args):
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
    gpt2_path = find_detail_model(config.base_path, args.activation)

    # collect all sentences in training set
    random.seed(999)
    dataset = {s: load_dataset(p) for s, p in data_path.items()}
    real = sum([[d for d in v] for v in dataset['train']], [])
    real = [' '.join([str(i) for i in tokenizer.encode(s)]) for s in real]

    ########################### case of ours ###########################
    if model not in ['seqgan', 'rankgan', 'maligan', 'mle', 'pgbleu']:
        torch.manual_seed(999)
        # load seed interpretation model
        gpt2 = InterpretationModel(config, pad_idx=tokenizer.pad_token_id, device=device).to(device)
        check_point = torch.load(gpt2_path, map_location=device)
        gpt2.load_state_dict(check_point['model']['gpt2'])

        # make fixed noise and load the model
        np.random.seed(999)
        fixed_noise = make_noise(n_samples, config.noise_init_size)
        fixed_noise = [fixed_noise[i * 100:(i + 1) * 100] for i in range(int(n_samples / 100))]
        generator = Generator(config).to(device)
        base_path = './model/' + model
        model_path = [base_path + '/' + model + '_' + str(i + 1) + '.pt' for i in range(len(os.listdir(base_path)) - 1)]

        # check repeat rate
        for fake_p in tqdm(model_path):
            repeat_num = 0
            check_point = torch.load(fake_p, map_location=device)
            generator.load_state_dict(check_point['model']['generator'])
            generator.eval()
            fake = sum([[greedy_search(gpt2.model, tokenizer, generator(n.to(device))[j].unsqueeze(0), config.max_len,
                                       config.activation, device, True)[0, config.max_len - 1:] for j in range(n.size(0))] for n
                        in fixed_noise], [])
            fake = [' '.join([str(i) for i in d.cpu().tolist()]) for d in fake]

            for s in fake:
                if s in real and len(s.split()) > 10:
                    repeat_num += 1

            dup, div = 1 - repeat_num / len(fake), len(set(fake)) / len(fake)
            repeat_rate[fake_p] = (dup, div, dup * div, 2 * dup * div / (dup + div))
            print(repeat_rate[fake_p])

        # save the repeat rate
        save_path = 'score/' + model + '_' + score_type.upper() + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(repeat_rate, f)

    ########################### case of others ###########################
    elif model in ['seqgan', 'rankgan', 'maligan', 'mle', 'pgbleu']:
        names = {'seqgan': 'SeqGAN', 'rankgan': 'RankGAN', 'maligan': 'MaliGAN', 'mle': 'MLE', 'pgbleu': 'PG_Bleu'}
        real = []
        with open('model/' + model + '/real_train_dailydialog.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                real.append(line)

        model_path = ['model/' + model + '/generator_sample' + str(i) + '.txt' for i in
                      range(len(list(filter(lambda x: x.startswith('generator'), os.listdir('model/' + model)))))]
        for fake_p in tqdm(model_path):
            fake, repeat_num = [], 0
            with open(fake_p, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    fake.append(line)

            for s in fake:
                if s in real and len(s.split()) > 10:
                    repeat_num += 1

            dup, div = 1 - repeat_num / len(fake), len(set(fake)) / len(fake)
            repeat_rate[fake_p] = (dup, div, dup * div, 2 * dup * div / (dup + div))
            print(repeat_rate[fake_p])

        # save the repeat rate
        save_path = 'score/' + names[model] + '_' + score_type.upper() + '.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(repeat_rate, f)


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
    elif args.type.lower() == 'sdr':
        sdr(args)
    else:
        print('FBD, MSJ, SDR are possible, please check...')
        assert AssertionError
