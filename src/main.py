import os
import json
import pickle
from argparse import ArgumentParser

import torch

from utils.config import Config
from train import TESGANTrainer, InterpretationTrainer



def main(config_path: Config, args: ArgumentParser):
    # defining device, config path, and base path
    device = torch.device('cuda:0') if args.device == 'gpu' else torch.device('cpu')
    print('Using {}'.format(device))

    if args.mode == 'train':
        config = Config(config_path)
        base_path = config.base_path

        # making neccessary folders
        os.makedirs(base_path + 'model', exist_ok=True)
        os.makedirs(base_path + 'history', exist_ok=True)
        os.makedirs(base_path + 'score', exist_ok=True)
        os.makedirs(base_path + 'syn', exist_ok=True)

        # defining loss history data path
        config.history_data_path = base_path + 'history/' + config.history_data_name + '.pkl'
        history_data_path = config.history_data_path

        # making model related files and folder
        model_folder = base_path + 'model/' + config.model_name
        model_json_path = model_folder + '/' + config.model_name + '.json'
        config.model_path = model_folder + '/' + config.model_name
        os.makedirs(model_folder, exist_ok=True)

        with open(model_json_path, 'w') as f:
            json.dump(config.__dict__, f)

        # types of training model
        if config.model.lower() == 'tesgan':
            config.model = 'tesgan'
            trainer = TESGANTrainer(config, device, args.mode, args.interp)
        elif config.model.lower() == 'interpretation':
            config.model = 'interpretation'
            trainer = InterpretationTrainer(config, device, args.mode)

        # training
        print('Start training...\n')
        try:
            train_loss_history, val_loss_history, score_history, best_scores = trainer.train()
            history_data = {'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history,
                            'score_history': score_history, 'best_scores': best_scores}

            # saving history data
            print('Saving the loss related data...')
            with open(history_data_path, 'wb') as f:
                pickle.dump(history_data, f)
        except TypeError:
            print('TESGAN training finished..')

    elif args.mode == 'syn':
        # path needed
        model_name = args.tesgan_name
        folder_name = model_name[:model_name.rfind('_')]
        model_path = 'model/' + folder_name + '/' + model_name
        assert os.path.isfile(model_path)

        # config
        config_path = model_path[:model_path.rfind('_')] + '.json'
        config = Config(config_path)
        config.syn_len = args.syn_len

        # synthesizing
        trainer = TESGANTrainer(config, device, args.mode, args.interp)
        trainer.syn(model_path, folder_name)


if __name__ == '__main__':
    path = os.path.realpath(__file__)
    path = path[:path.rfind('/') + 1] + 'config.json'

    parser = ArgumentParser()
    parser.add_argument('-d', '--device', type=str, required=True, choices=['cpu', 'gpu'])
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'syn'])
    parser.add_argument('-n', '--tesgan_name', type=str, required=False)
    parser.add_argument('--interp', type=str, required=False)
    parser.add_argument('--syn_len', default=16, type=int, required=False)
    args = parser.parse_args()

    main(path, args)
