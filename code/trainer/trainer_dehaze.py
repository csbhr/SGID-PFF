import decimal
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import data_utils
from trainer.trainer import Trainer
import torch.optim as optim
from loss import gradient_loss


class Trainer_Dehaze(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_Dehaze, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer_Dehaze")
        device = 'cpu' if args.cpu else 'cuda'
        self.grad_loss = gradient_loss.Gradient_Loss(device=device)
        self.l1_loss = nn.L1Loss().to(device)

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer = optim.Adam([{"params": self.model.get_model().dehaze.parameters()},
                                {"params": self.model.get_model().pre_dehaze.parameters(), "lr": 1e-5}],
                               **kwargs)
        return optimizer

    def train(self):
        print("Now training")
        self.scheduler.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))
        self.model.train()
        self.ckp.start_log()

        for batch, (input, gt, _) in enumerate(self.loader_train):

            input = input.to(self.device)
            gt = gt.to(self.device)

            pre_est_J, output, _, _, mid_loss = self.model(input)

            self.optimizer.zero_grad()
            loss, loss_log = self.loss(output, gt)

            grad_loss = self.grad_loss(output, gt)
            loss = loss + 0.1 * grad_loss
            loss_log['grad'] = grad_loss.item()
            loss_log['Total'] = loss.item()

            refer_loss = self.l1_loss(pre_est_J, gt)
            loss = loss + refer_loss
            loss_log['refer'] = refer_loss.item()
            loss_log['Total'] = loss.item()

            if mid_loss:  # mid loss is the loss during the model
                effective_mid_loss = self.args.mid_loss_weight * mid_loss
                loss = loss + effective_mid_loss
                loss_log['others'] = effective_mid_loss.item()
                loss_log['Total'] = loss.item()
            loss.backward()
            self.optimizer.step()

            self.ckp.report_log(loss_log)

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : {}'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.display_loss(batch)
                ))

            if (batch + 1) % self.args.max_iter_save == 0:
                self.model.save_now_model(apath=self.ckp.dir,
                                          flag='{}_{}'.format(epoch - 1, (batch + 1) // self.args.max_iter_save))

        self.ckp.end_log(len(self.loader_train))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (input, gt, filename) in enumerate(tqdm_test):

                filename = filename[0]
                input = input.to(self.device)
                gt = gt.to(self.device)

                _, output, trans, air, _ = self.model(input)

                PSNR = data_utils.calc_psnr(gt, output, rgb_range=self.args.rgb_range, )
                self.ckp.report_log(PSNR, train=False)

                if self.args.save_images:
                    gt, input, output, trans, air = data_utils.postprocess(gt, input, output, trans, air,
                                                                           rgb_range=self.args.rgb_range)
                    combine1 = np.concatenate((input, output, gt), axis=1)
                    combine2 = np.concatenate((trans, air, air), axis=1)
                    combine = np.concatenate((combine1, combine2), axis=0)
                    save_list = [combine]
                    self.ckp.save_images(filename, save_list)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.args.data_test, self.ckp.psnr_log[-1],
                best[0], best[1] + 1))
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
