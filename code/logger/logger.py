import torch
import imageio
import numpy as np
import os
import datetime

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Logger:
    def __init__(self, args, init_loss_log):
        self.args = args
        self.psnr_log = torch.Tensor()
        self.loss_log = {}
        for key in init_loss_log.keys():
            self.loss_log[key] = torch.Tensor()

        if args.load == '.':
            if args.save == '.':
                args.save = datetime.datetime.now().strftime('%Y%m%d_%H:%M')
            self.dir = args.experiment_dir + args.save
        else:
            self.dir = args.experiment_dir + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.loss_log = torch.load(self.dir + '/loss_log.pt')
                self.psnr_log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.psnr_log)))

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            if not os.path.exists(self.dir + '/model'):
                os.makedirs(self.dir + '/model')
        if not os.path.exists(self.dir + '/result/' + self.args.data_test):
            print("Creating dir for saving images...", self.dir + '/result/' + self.args.data_test)
            os.makedirs(self.dir + '/result/' + self.args.data_test)

        print('Save Path : {}'.format(self.dir))

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write('From epoch {}...'.format(len(self.psnr_log)) + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')

    def save(self, trainer, epoch, is_best):
        trainer.model.save(self.dir, epoch, is_best)
        torch.save(self.loss_log, os.path.join(self.dir, 'loss_log.pt'))
        torch.save(self.psnr_log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))
        self.plot_loss_log(epoch)
        self.plot_psnr_log(epoch)

    def save_images(self, filename, save_list):
        dirname = '{}/result/{}'.format(self.dir, self.args.data_test)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        filename = '{}/{}'.format(dirname, filename)
        if self.args.task == '.':
            postfix = ['combine']
        else:
            postfix = ['combine']
        for img, post in zip(save_list, postfix):
            imageio.imwrite('{}_{}.png'.format(filename, post), img)

    def start_log(self, train=True):
        if train:
            for key in self.loss_log.keys():
                self.loss_log[key] = torch.cat((self.loss_log[key], torch.zeros(1)))
        else:
            self.psnr_log = torch.cat((self.psnr_log, torch.zeros(1)))

    def report_log(self, item, train=True):
        if train:
            for key in item.keys():
                self.loss_log[key][-1] += item[key]
        else:
            self.psnr_log[-1] += item

    def end_log(self, n_div, train=True):
        if train:
            for key in self.loss_log.keys():
                self.loss_log[key][-1].div_(n_div)
        else:
            self.psnr_log[-1].div_(n_div)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for key in self.loss_log.keys():
            log.append('[{}: {:.4f}]'.format(key, self.loss_log[key][-1] / n_samples))
        return ''.join(log)

    def plot_loss_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for key in self.loss_log.keys():
            label = '{} Loss'.format(key)
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.loss_log[key].numpy())
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(self.dir, 'loss_{}.pdf'.format(key)))
            plt.close(fig)

    def plot_psnr_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('PSNR Graph')
        plt.plot(axis, self.psnr_log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'psnr.pdf'))
        plt.close(fig)

    def done(self):
        self.log_file.close()
