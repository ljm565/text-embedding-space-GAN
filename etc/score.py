import os
import sys
import random
import pickle
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.append('src/')

import torch

from utils.utils_func import *
from bert_distances import FBD
from utils.config import Config
from tokenizer import Tokenizer
from language_modeling import LanguageModeling
from multiset_distances import MultisetDistances
from data_synthesis_ratio import DataSynthesisRatio
from models import InterpretationModel, Generator



def main(dataset, args):
    # path needed
    print('Initializing...')
    model = args.model
    interp = args.interp
    trainset_path = "./data/dailydialog/processed/" + dataset.lower() + ".train"
    testset_path = "./data/dailydialog/processed/" + dataset.lower() + ".test"
    data_path = {'train': trainset_path, 'test': testset_path}
    config_path = 'model/' + model + '/' + model + '.json'

    # init
    n_samples = 1000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    try:
        config = Config(config_path)
    except:
        config = Config('src/config.json')
    tokenizer = Tokenizer(config).pretrained_tokenizer
    config.vocab_size = len(tokenizer)

    # collect real text
    random.seed(999)
    dataset = {s: load_dataset(p) for s, p in data_path.items()}
    train_real = sum([[d for d in v] for v in dataset['train']], [])
    train_real = [' '.join([str(i) for i in tokenizer.encode(s)]) for s in train_real]
    test_real = sum([[d for d in v] for v in dataset['test']], [])
    test_real = random.sample(test_real, n_samples)
    msj_real = [(''.join([c for c in s])).split() for s in test_real]
    
    # init scores
    scores = {}
    fbd = FBD(references=test_real, model_name="bert-base-uncased", bert_model_dir="/tmp/Bert/")
    msj = MultisetDistances(references=msj_real)
    dsr = DataSynthesisRatio(reference=train_real)
    lm = LanguageModeling(device=device)

    print('Scoring...')
    if model not in ['seqgan', 'rankgan', 'maligan', 'mle', 'pgbleu']:
        # load seed interpretation model
        torch.manual_seed(999)
        gpt2 = InterpretationModel(config, device=device).to(device)
        gpt2_path = find_detail_model(config.base_path, interp)
        check_point = torch.load(gpt2_path, map_location=device)
        gpt2.load_state_dict(check_point['model']['gpt2'])

        # load generator and fixed noise
        np.random.seed(999)
        fixed_noise = make_noise(n_samples, config.noise_init_size)
        fixed_noise = [fixed_noise[i * 100:(i + 1) * 100] for i in range(int(n_samples / 100))]
        generator = Generator(config, device).to(device)
        base_path = './model/' + model
        model_path = [base_path + '/' + model + '_' + str(i + 1) + '.pt' for i in range(len(os.listdir(base_path)) - 1)]
        
        for p in tqdm(model_path):
            scores[p] = {}
            check_point = torch.load(p, map_location=device)
            generator.load_state_dict(check_point['model']['generator'])
            generator.eval()
            
            fake_tok = sum([[greedy_search(gpt2.model, tokenizer, generator(n.to(device))[j].unsqueeze(0), config.max_len, config.activation, device, True)[0, config.max_len - 1:] 
                               for j in range(n.size(0))] for n in fixed_noise], [])
            fake_text = [tokenizer.decode(f) for f in fake_tok]

            msj_fake = [(''.join([c for c in s])).split() for s in fake_text]
            dsr_fake = [' '.join([str(i) for i in d.cpu().tolist()]) for d in fake_tok]

            scores[p]['fbd'] = fbd.get_score(sentences=fake_text)
            scores[p]['msj'] = msj.get_jaccard_score(sentences=msj_fake)
            scores[p]['dsr'] = dsr.get_dsr(sentences=dsr_fake)
            scores[p]['lm'] = lm.get_lm(sentences=fake_text)
            print(scores)
        # save the data
        save_path = 'score/' + model + '_score.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(scores, f)


    elif model in ['seqgan', 'rankgan', 'maligan', 'mle', 'pgbleu']:
        names = {'seqgan': 'SeqGAN', 'rankgan': 'RankGAN', 'maligan': 'MaliGAN', 'mle': 'MLE', 'pgbleu': 'PG_Bleu'}
        model_path = ['model/' + model + '/generator_sample' + str(i) + '.txt' for i in
                      range(len(list(filter(lambda x: x.startswith('generator'), os.listdir('model/' + model)))))]

        # for dsr
        trainset = []
        with open('model/' + model + '/real_train_dailydialog.txt', 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            trainset.append(line)

        # compute metrics
        for p in model_path:
            scores[p] = {}
            fake, dsr_fake, repeat_num = [], [], 0
            with open(p, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()

                dsr_fake.append(line)
                line = [int(c) for c in line.split(' ')]
                
                for j in range(len(line)):
                    if line[j] == tokenizer.pad_token_id:
                        line = line[:j]
                        break
                fake.append(tokenizer.decode(line[:j]))
            
            fake = random.sample(fake, n_samples)
            msj_fake = [(''.join([c for c in s])).split() for s in fake]

            for s in dsr_fake:
                if s in trainset and len(s.split()) > 10:
                    repeat_num += 1
            dup, div = 1 - repeat_num / len(dsr_fake), len(set(dsr_fake)) / len(dsr_fake)
            
            scores[p]['fbd'] = fbd.get_score(sentences=fake)
            scores[p]['msj'] = msj.get_jaccard_score(sentences=msj_fake)
            scores[p]['dsr'] = (dup, div, dup * div, 2 * dup * div / (dup + div))

            with open(p, 'r') as f:
                lines = f.readlines()
            
            random.seed(999)
            lm_fake = random.sample(lines, n_samples)
            lm_fake = [tokenizer.decode(list(filter(lambda x: x != 50259, map(int, d.strip().split())))) for d in lm_fake]
            scores[p]['lm'] = lm.get_lm(sentences=lm_fake)
        
        # save the data
        save_path = 'score/' + names[model] + '_score.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(scores, f)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-a', '--activation', default='sigmoid', type=str, required=False,
                        choices=['sigmoid', 'tanh', 'none'])
    parser.add_argument('--interp', default=None, type=str, required=False)
    args = parser.parse_args()

    assert os.path.isdir('model/' + args.model)
    main('dailydialog', args)
