import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time

from utils.config import Config
from tokenizer import Tokenizer
from utils.utils_func import *
from utils.utils_data import DLoader4TESGAN, DLoader4Interpretation
from models import InterpretationModel, Generator, Discriminator_BERT, Discriminator_LSTM
import torch.nn.functional as F


class Trainer:
    def __init__(self, config: Config, device: str, state: str):
        self.config = config
        self.device = device
        self.state = state

        # paths
        self.data_path = {'train': self.config.trainset_path, 'val': self.config.valset_path,
                          'test': self.config.testset_path}
        self.model_path = self.config.model_path

        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.max_len = self.config.max_len

        # define tokenizer
        self.tokenizer = Tokenizer(self.config).pretrained_tokenizer
        self.config.vocab_size = len(self.tokenizer)

        # dataloaders
        DLoader = DLoader4Interpretation if self.config.model == 'interpretation' else DLoader4TESGAN
        self.dataset = {s: DLoader(load_dataset(p), self.tokenizer, self.config) for s, p in self.data_path.items()}
        self.dataloaders = {
            s: DataLoader(d, self.batch_size, shuffle=True) if s == 'train' or s == 'val' else DataLoader(d,
                                                                                                          self.batch_size,
                                                                                                          shuffle=False)
            for s, d in self.dataset.items()}

        # model and others
        self.pad_idx = self.tokenizer.pad_token_id
        self.interpretationModel = InterpretationModel(self.config, pad_idx=self.pad_idx, device=self.device).to(
            self.device)
        if self.config.model == 'tesgan':
            self.detail_model = find_detail_model(self.config.base_path, self.config.activation)
            self.check_point = torch.load(self.detail_model, map_location=self.device)
            self.interpretationModel.load_state_dict(self.check_point['model']['gpt2'])
            for p in self.interpretationModel.parameters():
                p.requires_grad = False

            self.generator = Generator(self.config).to(self.device)
            self.generator.apply(weights_init)
            if self.state == 'train':
                self.discriminator_bert = Discriminator_BERT(self.config).to(self.device)
                self.discriminator_lstm = Discriminator_LSTM().to(self.device)
                self.discriminator_bert.apply(weights_init)

        if self.state == 'train':
            self.lm_loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            self.bce_loss = nn.BCELoss()
            self.mae_loss = nn.L1Loss()
            self.mse_loss = nn.MSELoss()
            self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')

            self.interpretationModel_optimizer = optim.Adam(self.interpretationModel.parameters(), lr=0.001)
            if self.config.model == 'tesgan':
                self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
                self.discriminator_bert_optimizer = optim.Adam(self.discriminator_bert.parameters(), lr=0.0005,
                                                               betas=(0.5, 0.999))
                self.discriminator_lstm_optimizer = optim.Adam(self.discriminator_lstm.parameters(), lr=0.001)


class TESGANTrainer(Trainer):
    def __init__(self, config: Config, device: str, state: str):
        super(TESGANTrainer, self).__init__(config, device, state)

    def train(self):
        # for testing
        fixed_test_noise = make_noise(256, self.config.noise_init_size).to(self.device)

        # training starts
        for epoch in range(self.epochs):
            start = time.time()
            print(epoch + 1, '/', self.epochs)
            print('-' * 10)
            for phase in ['train', 'val']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    self.interpretationModel.eval()
                    self.generator.train()
                    self.discriminator_bert.train()
                    self.discriminator_lstm.train()
                else:
                    self.interpretationModel.eval()
                    self.generator.eval()
                    self.discriminator_bert.eval()
                    self.discriminator_lstm.eval()

                # initialize metric
                D_total_loss, G_total_loss = 0, 0
                total_ssd_r_acc, total_ssd_f_acc, total_sod_r_acc, total_sod_f_acc = 0, 0, 0, 0
                total_g_ssd_acc, total_g_sod_acc = 0, 0

                for i, (input) in enumerate(self.dataloaders[phase]):
                    b_size = input.size(0)
                    input = input.to(self.device)
                    self.interpretationModel_optimizer.zero_grad()
                    self.generator_optimizer.zero_grad()
                    self.discriminator_bert_optimizer.zero_grad()
                    self.discriminator_lstm_optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        # make real seed
                        real_data, vocab_output = self.interpretationModel(input)

                        ########################################### discriminator ###############################################
                        ##################
                        # real seed part #
                        ##################
                        target = torch.ones(b_size, dtype=torch.long).to(self.device)

                        # calculate ssd
                        ssd_r_loss, acc = self.discriminator_bert(real_data, target)
                        ssd_r_acc = torch.sum(torch.argmax(acc, dim=-1) == target) / b_size

                        # calculate sod
                        lstm_out = self.discriminator_lstm(real_data)
                        sod_r_loss = self.bce_loss(lstm_out, target.float())
                        sod_r_acc = torch.sum(torch.where(lstm_out > 0.5, 1, 0) == target) / b_size

                        ##################
                        # fake seed part #
                        ##################
                        target = torch.zeros(b_size, dtype=torch.long).to(self.device)
                        fake_seed = make_noise(b_size, self.config.noise_init_size)
                        fake_data = self.generator(fake_seed.to(self.device))

                        # calculate ssd
                        ssd_f_loss, acc = self.discriminator_bert(fake_data.detach(), target)
                        ssd_f_acc = torch.sum(torch.argmax(acc, dim=-1) == target) / b_size

                        # calculate sod
                        lstm_out = self.discriminator_lstm(fake_data.detach())
                        sod_f_loss = self.bce_loss(lstm_out, target.float())
                        sod_f_acc = torch.sum(torch.where(lstm_out > 0.5, 1, 0) == target) / b_size
                        D_loss = ssd_r_loss + ssd_f_loss + sod_r_loss + sod_f_loss

                        if phase == 'train' and epoch % 2 == 0:
                            D_loss.backward()
                            self.discriminator_bert_optimizer.step()
                            self.discriminator_lstm_optimizer.step()
                        ########################################################################################################

                        ############################################## generator ###############################################
                        for _ in range(self.config.g_train_num_per_epoch):
                            # make fake seed
                            fake_data = self.generator(make_noise(b_size, self.config.noise_init_size).to(self.device))
                            target = torch.ones(b_size, dtype=torch.long).to(self.device)

                            # calculate SSD
                            ssd, acc = self.discriminator_bert(fake_data, target)

                            # calculate SOD
                            lstm_out = self.discriminator_lstm(fake_data)
                            sod = self.bce_loss(lstm_out, target.float())

                            # calculate SFP
                            m1, m2 = torch.mean(real_data, dim=-1), torch.mean(fake_data, dim=-1)
                            m3, m4 = torch.mean(m1, dim=-1), torch.mean(m2, dim=-1)
                            m5, m6 = torch.mean(real_data), torch.mean(fake_data)
                            sfp = self.mse_loss(m2, m1) + self.mse_loss(m4, m3) + self.mse_loss(m6, m5) + self.mae_loss(
                                fake_data, real_data)

                            # calculate SDP
                            _, vocab_output_fake = self.interpretationModel(fake_data, True)
                            vocab_input = F.log_softmax(vocab_output_fake, dim=-1)
                            vocab_trg = F.softmax(vocab_output, dim=-1)
                            sdp = self.kldiv_loss(vocab_input, vocab_trg)

                            G_loss = ssd + sod + sdp + sfp
                            g_ssd_acc = torch.sum(torch.argmax(acc, dim=-1) == target) / b_size
                            g_sod_acc = torch.sum(torch.where(lstm_out > 0.5, 1, 0) == target) / b_size

                            if phase == 'train':
                                G_loss.backward()
                                self.generator_optimizer.step()
                        ########################################################################################################

                    # calculate loss and acc
                    D_total_loss += D_loss.item() * b_size
                    G_total_loss += G_loss.item() * b_size
                    total_ssd_r_acc += ssd_r_acc.item() * b_size
                    total_ssd_f_acc += ssd_f_acc.item() * b_size
                    total_sod_r_acc += sod_r_acc.item() * b_size
                    total_sod_f_acc += sod_f_acc.item() * b_size
                    total_g_ssd_acc += g_ssd_acc.item() * b_size
                    total_g_sod_acc += g_sod_acc.item() * b_size

                    if i % 50 == 0:
                        print('-' * 100)
                        print('Epoch {}: {}/{} step'.format(epoch + 1, i, len(self.dataloaders[phase])))
                        print(
                            'G_loss: {:.3f} (ssd: {:.3f}, sod: {:.3f}, sdp: {:.3f}, sfp: {:.3f}), D_loss: {:.3f} (ssd_r: {:.3f}, ssd_f: {:.3f}, sod_r: {:.3f}, sod_f: {:.3f})'.format(
                                G_loss.item(), ssd.item(), sod.item(), sdp.item(), sfp.item(), D_loss.item(),
                                ssd_r_loss.item(), ssd_f_loss.item(), sod_r_loss.item(), sod_f_loss.item()))
                        print(
                            'G_ssd_acc: {:.3f}, G_sod_acc: {:.3f}, D_ssd_r_acc: {:.3f}, D_ssd_f_acc: {:.3f}, D_sod_r_acc: {:.3f}, D_sod_f_acc: {:.3f}'.format(
                                g_ssd_acc.item(), g_sod_acc.item(), ssd_r_acc.item(), ssd_f_acc.item(),
                                sod_r_acc.item(), sod_f_acc.item()))

                epoch_D_loss = D_total_loss / len(self.dataloaders[phase].dataset)
                epoch_G_loss = G_total_loss / len(self.dataloaders[phase].dataset)
                total_ssd_r_acc = total_ssd_r_acc / len(self.dataloaders[phase].dataset)
                total_ssd_f_acc = total_ssd_f_acc / len(self.dataloaders[phase].dataset)
                total_sod_r_acc = total_sod_r_acc / len(self.dataloaders[phase].dataset)
                total_sod_f_acc = total_sod_f_acc / len(self.dataloaders[phase].dataset)
                total_g_ssd_acc = total_g_ssd_acc / len(self.dataloaders[phase].dataset)
                total_g_sod_acc = total_g_sod_acc / len(self.dataloaders[phase].dataset)
                print(
                    '{} loss: - G_loss: {:.3f}, D_loss: {:.3f}, D_ssd_r_acc: {:.3f}, D_ssd_f_acc: {:.3f}, D_sod_r_acc: {:.3f}, D_sod_f_acc: {:.3f}, G_sdd_acc: {:.3f}, G_sod_acc: {:.3f}' \
                    .format(phase, epoch_G_loss, epoch_D_loss, total_ssd_r_acc, total_ssd_f_acc, total_sod_r_acc,
                            total_sod_f_acc, total_g_ssd_acc, total_g_sod_acc))

                if phase == 'val':
                    # print examples
                    for _ in range(3):
                        idx = np.random.randint(b_size)
                        input_ids = input[idx].unsqueeze(0).detach()
                        in_l = input_ids.size(1)

                        print(self.tokenizer.decode(input_ids[0]))
                        greedy_output = self.interpretationModel.model.generate(input_ids, max_length=50)

                        print("Output:\n" + 100 * '-')
                        print(self.tokenizer.decode(greedy_output[0, in_l:]))

                        print('-' * 100)
                        i = greedy_search(self.interpretationModel.model, self.tokenizer, input_ids, self.max_len,
                                          self.config.activation, self.device)
                        print(self.tokenizer.decode(i[0, in_l:]))

                        print('-' * 100)
                        i = greedy_search(self.interpretationModel.model, self.tokenizer, fake_data[idx].unsqueeze(0),
                                          self.max_len, self.config.activation, self.device, True)
                        print(self.tokenizer.decode(i[0, in_l - 1:]))
                        print()

                    fake_s = self.generator(fixed_test_noise)
                    for j in range(fake_s.size(0)):
                        i = greedy_search(self.interpretationModel.model, self.tokenizer, fake_s[j].unsqueeze(0),
                                          self.max_len, self.config.activation, self.device, True)
                        print(str(j) + ': ' + self.tokenizer.decode(i[0, in_l - 1:]))
                    print()

            # save the last model
            save_checkpoint(self.model_path + '_' + str(epoch + 1) + '.pt', {'generator': self.generator.state_dict(),
                                                                             'discriminator': self.discriminator_bert.state_dict()},
                            {'generator': self.generator_optimizer.state_dict(),
                             'discriminator': self.discriminator_bert_optimizer.state_dict()})
            print("training time :", time.time() - start)
            print('\n' * 3)

    def syn(self, model_path, file_name):
        # interpretation model
        self.interpretationModel.eval()

        # make fixed noise and load the model
        np.random.seed(999)
        torch.manual_seed(999)
        n_samples = 1000
        fixed_noise = make_noise(n_samples, self.config.noise_init_size)
        fixed_noise = [fixed_noise[i * 100:(i + 1) * 100] for i in range(int(n_samples / 100))]
        generator = Generator(self.config).to(self.device)

        # synthesize sentences
        check_point = torch.load(model_path, map_location=self.device)
        generator.load_state_dict(check_point['model']['generator'])
        generator.eval()
        fake = sum([[self.tokenizer.decode(
            greedy_search(self.interpretationModel.model, self.tokenizer, generator(n.to(self.device))[j].unsqueeze(0),
                          self.config.max_len, self.config.activation, self.device, True)[0, self.config.max_len - 1:]) for j in
                     range(n.size(0))] for n in fixed_noise], [])
        with open('syn/' + file_name + '.txt', 'w') as f:
            for s in fake:
                f.write(s + '\n')


class InterpretationTrainer(Trainer):
    def __init__(self, config: Config, device: str, state: str):
        super(InterpretationTrainer, self).__init__(config, device, state)

    def train(self):
        best_ppl, best_bleu_2, best_bleu_4, best_nist_2, best_nist_4 = float('inf'), 0, 0, 0, 0
        loss_key = ['interp_loss']
        score_key = ['bleu-2', 'bleu-4', 'nist-2', 'nist-4']
        train_loss_history, val_loss_history, score_history = {k: [] for k in loss_key}, {k: [] for k in loss_key}, {
            k: [] for k in score_key}

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch + 1, '/', self.epochs)
            print('-' * 10)
            for phase in ['train', 'val']:
                print('Phase: {}'.format(phase))
                losses = {k: [] for k in loss_key}
                if phase == 'train':
                    self.interpretationModel.train()
                else:
                    self.interpretationModel.eval()
                    val_s = []

                for i, (input, src_len, data_len) in enumerate(self.dataloaders[phase]):
                    b_size = input.size(0)
                    input = input.to(self.device)
                    self.interpretationModel_optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        _, lm_output = self.interpretationModel(input)
                        interp_loss = self.lm_loss(lm_output[:, :-1, :].reshape(-1, lm_output.size(-1)),
                                                   input[:, 1:].reshape(-1))
                        if phase == 'train':
                            interp_loss.backward()
                            self.interpretationModel_optimizer.step()

                            # save the losses and print the losses
                    losses['interp_loss'].append(interp_loss.item() * b_size)
                    if i % self.config.print_log_interv == 0:
                        print('Epoch {}: {}/{} step - interp loss: {:.3f}'.format(epoch + 1, i,
                                                                                  len(self.dataloaders[phase]),
                                                                                  losses['interp_loss'][-1] / b_size))

                    # save the target, output sentences
                    if phase == 'val':
                        val_s = save_val_s(b_size, val_s, lm_output, input, src_len, data_len, self.tokenizer)

                # print phase logs
                epoch_interp_loss = sum(losses['interp_loss']) / len(self.dataloaders[phase].dataset)
                print('{} loss: - interp loss: {:.3f}'.format(phase, epoch_interp_loss))

                if phase == 'train':
                    train_loss_history = save_loss_history(train_loss_history, losses,
                                                           len(self.dataloaders[phase].dataset))
                    ppl = cal_ppl(epoch_interp_loss)
                    logger(phase + ' ppl', ppl)
                    print()

                if phase == 'val':
                    val_loss_history = save_loss_history(val_loss_history, losses, len(self.dataloaders[phase].dataset))
                    score_history = save_score_history(score_history, val_s)

                    # print and calculate scores
                    ppl, bleu_2, bleu_4, nist_2, nist_4 = cal_ppl(epoch_interp_loss), score_history['bleu-2'][-1], \
                                                          score_history['bleu-4'][-1], score_history['nist-2'][-1], \
                                                          score_history['nist-4'][-1]
                    logger(phase + ' ppl: ', ppl)
                    logger(phase + ' bleu-2: ', bleu_2)
                    logger(phase + ' bleu-4: ', bleu_4)
                    logger(phase + ' nist-2: ', nist_2)
                    logger(phase + ' nist-4: ', nist_4)
                    print()

                    # print examples
                    tmp = 3
                    while tmp != 0:
                        idx = np.random.randint(b_size)
                        tmp -= 1
                        st, tr = src_len[idx], data_len[idx]
                        src = self.tokenizer.decode(input[idx, :st].detach().cpu().tolist())
                        gt = self.tokenizer.decode(input[idx, st:tr].detach().cpu().tolist())
                        pred = self.tokenizer.decode(
                            torch.argmax(lm_output[idx, st - 1:tr - 1].detach().cpu(), dim=1).tolist())
                        print('src : {}'.format(src))
                        print('gt  : {}'.format(gt))
                        print('pred: {}'.format(pred))
                        print()

                    # save the model
                    best_ppl = save_model(self.model_path, ppl, best_ppl, 'ppl', self.interpretationModel,
                                          self.interpretationModel_optimizer, epoch)
                    best_bleu_2 = save_model(self.model_path, bleu_2, best_bleu_2, 'bleu2', self.interpretationModel,
                                             self.interpretationModel_optimizer, epoch)
                    best_bleu_4 = save_model(self.model_path, bleu_4, best_bleu_4, 'bleu4', self.interpretationModel,
                                             self.interpretationModel_optimizer, epoch)
                    best_nist_2 = save_model(self.model_path, nist_2, best_nist_2, 'nist2', self.interpretationModel,
                                             self.interpretationModel_optimizer, epoch)
                    best_nist_4 = save_model(self.model_path, nist_4, best_nist_4, 'nist4', self.interpretationModel,
                                             self.interpretationModel_optimizer, epoch)

            # save the last model
            saving(self.model_path, 'last', self.interpretationModel, self.interpretationModel_optimizer, epoch)
            print("training time :", time.time() - start, 'sec')
            print('\n' * 3)

        logger('best ppl: ', best_ppl)
        logger('best bleu-2: ', best_bleu_2)
        logger('best bleu-4: ', best_bleu_4)
        logger('best nist-2: ', best_nist_2)
        logger('best nist-4: ', best_nist_4)
        best_scores = {'best_ppl': best_ppl, 'best_bleu-2': best_bleu_2, 'best_bleu-4': best_bleu_4,
                       'best_nist-2': best_nist_2, 'best_nist-4': best_nist_4}
        return train_loss_history, val_loss_history, score_history, best_scores
