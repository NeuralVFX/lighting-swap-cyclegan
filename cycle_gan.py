# TRAINING CLASS #
import math
import itertools
import time

from tqdm import tqdm
from torch.utils.data import *
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.nn as nn

from util import helpers as helper
from util import loaders as load
from models import networks as n


def mean_std_loss(f_real, f_fake):
    # Loop through features collected from real and fake discriminators, measure difference in std and mean
    std_losses = 0
    mean_losses = 0
    for real_feat, fake_feat in zip(f_real, f_fake):
        std_loss = ((real_feat.std(0) - fake_feat.std(0)) * (real_feat.std(0) - fake_feat.std(0))).mean()
        mean_loss = ((real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))).mean()
        std_losses += std_loss
        mean_losses += mean_loss
    return mean_losses, std_losses


############################################################################
# Train
############################################################################

class CycleGan:
    """
    Example usage if not using command line:

    params = {'dataset': 'wat_mai_amataros',
              'train_folder': 'train',
              'A': 'morning',
              'B': 'cloudy',
              'test_folder': 'test',
              'in_channels': 3,
              'batch_size': 7,
              'gen_filters': 256,
              'disc_filters': 512,
              'res_blocks': 9,
              'input_size': 128,
              'lr_disc': 1e-4,
              'lr_gen': 1e-4,
              'lr_cycle_mult': 2,
              'cycle_loss_A': 10,
              'cycle_loss_B': 10,
              'similar_distance': 10,
              'beta1': .5,
              'beta2': .999,
              'gen_layers': 2,
              'disc_layers': 3,
              'img_input_size':270,
              'img_output_size':128,
              'ids_a': [20, 60],
              'ids_b': [20, 70],
              'save_root': 'wat_mai}
    lgtSwap = CycleGan(params)
    lgtSwap.train()
    """

    def __init__(self, params):
        self.params = params
        self.model_dict = {}
        self.opt_dict = {}
        self.current_epoch = 0
        self.current_iter = 0
        self.current_cycle = 0

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5))])

        self.train_loader, data_len = load.data_load(f'data/{params["dataset"]}/{params["train_folder"]}/{params["A"]}',
                                                     f'data/{params["dataset"]}/{params["train_folder"]}/{params["B"]}',
                                                     transform, params["batch_size"], shuffle=True, cache=True,
                                                     cache_file=f'{params["dataset"]}_content_cache.pickle',
                                                     close=params["similar_distance"],
                                                     input_res=params["img_input_size"],
                                                     output_res=params["img_output_size"])

        self.set_lr_sched(params['train_epoch'], math.ceil(float(data_len) / float(params['batch_size'])),
                          params['lr_cycle_mult'])

        self.model_dict["G_A"] = n.Generator(layers=params["gen_layers"], filts=params["gen_filters"],
                                             channels=params["in_channels"], res_layers=params["res_blocks"])
        self.model_dict["G_B"] = n.Generator(layers=params["gen_layers"], filts=params["gen_filters"],
                                             channels=params["in_channels"], res_layers=params["res_blocks"])
        self.model_dict["D_A"] = n.Discriminator(layers=params["disc_layers"], filts=params["disc_filters"],
                                                 channels=params["in_channels"])
        self.model_dict["D_B"] = n.Discriminator(layers=params["disc_layers"], filts=params["disc_filters"],
                                                 channels=params["in_channels"])

        for i in self.model_dict.keys():
            self.model_dict[i].apply(helper.weights_init_normal)
            self.model_dict[i].cuda()
            self.model_dict[i].train()

        print('Networks Initialized')

        # setup losses #
        self.BCE_loss = nn.BCELoss()
        self.L1_loss = nn.L1Loss()

        # setup optimizers #
        self.opt_dict["G"] = optim.Adam(
            itertools.chain(self.model_dict["G_A"].parameters(), self.model_dict["G_B"].parameters()),
            lr=params['lr_gen'], betas=(params['beta1'], params['beta2']), weight_decay=.00001)
        self.opt_dict["D_A"] = optim.Adam(self.model_dict["D_A"].parameters(), lr=params['lr_disc'],
                                          betas=(params['beta1'], params['beta2']), weight_decay=.00001)
        self.opt_dict["D_B"] = optim.Adam(self.model_dict["D_B"].parameters(), lr=params['lr_disc'],
                                          betas=(params['beta1'], params['beta2']), weight_decay=.00001)

        # setup fake image pool #
        self.fakeA_pool = helper.ImagePool(50)
        self.fakeB_pool = helper.ImagePool(50)
        print('Losses and Pools Initialized')

        # setup history storage #
        self.losses = ['D_A_feat_loss', 'D_B_feat_loss', 'D_A_loss', 'D_B_loss', 'G_A_loss', 'G_B_loss', 'Cycle_A_loss',
                       'Cycle_B_loss']
        self.loss_batch_dict = {}
        self.loss_epoch_dict = {}
        self.train_hist_dict = {'per_epoch_ptimes': [], 'total_ptime': {}}

        for loss in self.losses:
            self.train_hist_dict[loss] = []
            self.loss_epoch_dict[loss] = []
            self.loss_batch_dict[loss] = []

    def load_state(self, filepath):
        # Load previously saved sate from disk, including models, optimizers and history
        state = torch.load(filepath)
        self.current_iter = state['iter'] + 1
        self.current_epoch = state['epoch'] + 1
        self.current_cycle = state['cycle'] + 1
        for i in self.model_dict.keys():
            self.model_dict[i].load_state_dict(state['models'][i])
        for i in self.opt_dict.keys():
            self.opt_dict[i].load_state_dict(state['optimizers'][i])
        self.train_hist_dict = state['train_hist']

    def save_state(self, filepath):
        # Save current state of all models, optimizers and history to disk
        out_model_dict = {}
        out_opt_dict = {}
        for i in self.model_dict.keys():
            out_model_dict[i] = self.model_dict[i].state_dict()
        for i in self.opt_dict.keys():
            out_opt_dict[i] = self.opt_dict[i].state_dict()
        model_state = {'iter': self.current_iter,
                       'cycle': self.current_cycle,
                       'epoch': self.current_epoch,
                       'models': out_model_dict,
                       'optimizers': out_opt_dict,
                       'train_hist': self.train_hist_dict}
        torch.save(model_state, filepath)
        return f'Saving State at Iter:{self.current_iter}'

    def lr_lookup(self):
        # Determine proper learning rate multiplier for this iter
        lr_mult = self.iter_stack[self.current_iter]
        save = self.current_iter in self.save_index
        return lr_mult, save

    def set_lr_sched(self, epochs, iters, mult):
        # Test implementation of warm restarts, makes training wat_mai dataset easier
        mult_iter = iters
        iter_stack = []
        save_index = []
        for a in range(epochs):
            iter_stack += [math.cos((x / mult_iter) * 3.14) * .5 + .5 for x in (range(int(mult_iter)))]
            mult_iter *= mult
            save_index.append(len(iter_stack) - 1)

        self.iter_stack = iter_stack
        self.save_index = save_index

        fig = plt.figure()
        plt.plot(self.iter_stack)
        plt.savefig(f'output/{self.params["save_root"]}_learning_rate_schedule.jpg')
        plt.ylabel('Learning Rate Schedule')
        plt.show()
        plt.close(fig)

    def display_history(self):
        # Draw history of losses, called at end of training
        fig = plt.figure()
        for key in self.losses:
            x = range(len(self.train_hist_dict[key]))
            if len(x) > 0:
                plt.plot(x, self.train_hist_dict[key], label=key)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'output/{self.params["save_root"]}_loss.jpg')
        plt.show()
        plt.close(fig)

    def train(self):
        # Train following learning rate schedule
        params = self.params
        done = False
        while not done:
            # clear last epochs losses
            for loss in self.losses:
                self.loss_epoch_dict[loss] = []

            epoch_start_time = time.time()
            num_iter = 0

            print(f"Sched Cycle:{self.current_cycle}, Sched Iter:{self.current_iter}, Sched Epoch:{self.current_epoch}")
            [print(f"Learning Rate({opt}): {self.opt_dict[opt].param_groups[0]['lr']}") for opt in
             self.opt_dict.keys()]

            for (real_a, real_b) in tqdm(self.train_loader):

                if self.current_iter > len(self.iter_stack) - 1:
                    done = True
                    self.display_history()
                    print('Hit End of Learning Schedule!')
                    break

                lr_mult, save = self.lr_lookup()
                self.opt_dict["D_A"].param_groups[0]['lr'] = lr_mult * params['lr_disc']
                self.opt_dict["D_B"].param_groups[0]['lr'] = lr_mult * params['lr_disc']
                self.opt_dict["G"].param_groups[0]['lr'] = lr_mult * params['lr_gen']

                real_a, real_b = Variable(real_a.cuda()), Variable(real_b.cuda())

                # traing generator
                self.opt_dict["G"].zero_grad()

                # generate fake b and discriminate
                fake_b = self.model_dict["G_A"](real_a)
                d_a_result, d_a_fake_feats = self.model_dict["D_A"](fake_b)
                self.loss_batch_dict['G_A_loss'] = self.BCE_loss(d_a_result,
                                                                 Variable(torch.ones(d_a_result.size()).cuda()))

                # reconstruct a
                rec_a = self.model_dict["G_B"](fake_b)
                self.loss_batch_dict['Cycle_A_loss'] = self.L1_loss(rec_a, real_a) * params['cycle_loss_A']

                # generate fake a and discriminate
                fake_a = self.model_dict["G_B"](real_b)
                d_b_result, d_b_fake_feats = self.model_dict["D_B"](fake_a)
                self.loss_batch_dict['G_B_loss'] = self.BCE_loss(d_b_result,
                                                                 Variable(torch.ones(d_b_result.size()).cuda()))
                # reconstruct b
                rec_b = self.model_dict["G_A"](fake_a)
                self.loss_batch_dict['Cycle_B_loss'] = self.L1_loss(rec_b, real_b) * params['cycle_loss_B']

                self.opt_dict["D_A"].zero_grad()
                self.opt_dict["D_B"].zero_grad()

                # discriminate real samples
                d_a_real, d_a_real_feats = self.model_dict["D_A"](real_b)
                d_b_real, d_b_real_feats = self.model_dict["D_B"](real_a)

                d_a_mean_loss, d_a_std_loss = mean_std_loss(d_a_real_feats, d_a_fake_feats)
                d_b_mean_loss, d_b_std_loss = mean_std_loss(d_b_real_feats, d_b_fake_feats)

                # calculate feature loss
                self.loss_batch_dict['D_A_feat_loss'] = d_a_std_loss + d_a_mean_loss
                self.loss_batch_dict['D_B_feat_loss'] = d_b_std_loss + d_b_mean_loss

                # addup generator a and b loss and step
                g_a_loss_total = self.loss_batch_dict['G_A_loss'] * .5 + self.loss_batch_dict['D_B_feat_loss'] * .5
                g_b_loss_total = self.loss_batch_dict['G_B_loss'] * .5 + self.loss_batch_dict['D_A_feat_loss'] * .5

                g_loss = g_a_loss_total + g_b_loss_total + self.loss_batch_dict['Cycle_A_loss'] + self.loss_batch_dict[
                    'Cycle_B_loss']
                g_loss.backward(retain_graph=True)
                self.opt_dict["G"].step()

                # train discriminator a
                d_a_real_loss = self.BCE_loss(d_a_real, Variable(torch.ones(d_a_real.size()).cuda()))

                fake_b = self.fakeB_pool.query(fake_b)
                d_a_fake, d_a_fake_feats = self.model_dict["D_A"](fake_b)
                d_a_fake_loss = self.BCE_loss(d_a_fake, Variable(torch.zeros(d_a_fake.size()).cuda()))

                # add up disc a loss and step
                self.loss_batch_dict['D_A_loss'] = (d_a_real_loss + d_a_fake_loss) * .5
                self.loss_batch_dict['D_A_loss'].backward()
                self.opt_dict["D_A"].step()

                # train discriminator b
                d_b_real_loss = self.BCE_loss(d_b_real, Variable(torch.ones(d_b_real.size()).cuda()))

                fake_a = self.fakeA_pool.query(fake_a)
                d_b_fake, d_b_fake_feats = self.model_dict["D_B"](fake_a)
                d_b_fake_loss = self.BCE_loss(d_b_fake, Variable(torch.zeros(d_b_fake.size()).cuda()))

                # add up disc b  loss and step
                self.loss_batch_dict['D_B_loss'] = (d_b_real_loss + d_b_fake_loss) * .5
                self.loss_batch_dict['D_B_loss'].backward()
                self.opt_dict["D_B"].step()

                # append all losses in loss dict
                [self.train_hist_dict[loss].append(self.loss_batch_dict[loss].data[0]) for loss in self.losses]
                [self.loss_epoch_dict[loss].append(self.loss_batch_dict[loss].data[0]) for loss in self.losses]

                if save:
                    save_str = self.save_state(f'output/{params["save_root"]}_{self.current_epoch}.json')
                    tqdm.write(save_str)
                    self.current_epoch += 1

                self.current_iter += 1
                num_iter += 1

            helper.show_test(self.model_dict['G_A'], self.model_dict['G_B'], params,
                             save=f'output/{params["save_root"]}_{self.current_cycle}.jpg')
            if not done:
                self.current_cycle += 1
                epoch_end_time = time.time()
                per_epoch_ptime = epoch_end_time - epoch_start_time
                self.train_hist_dict['per_epoch_ptimes'].append(per_epoch_ptime)
                print(f'Epoch:{self.current_epoch}, Epoch Time:{per_epoch_ptime}')
                [print(f'{loss}: {helper.mft(self.loss_epoch_dict[loss])}') for loss in self.losses]
