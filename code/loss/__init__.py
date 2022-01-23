from importlib import import_module
import torch
import torch.nn as nn


class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        device = torch.device('cpu' if args.cpu else 'cuda')

        self.loss = []
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss().to(device)
            elif loss_type == 'L1':
                loss_function = nn.L1Loss().to(device)
            elif loss_type == 'HEM':
                module = import_module('loss.hem_loss')
                loss_function = getattr(module, 'HEM')(device=device).to(device)
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')().to(device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found'.format(loss_type))

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            print('{:.3f} * {}'.format(float(weight), loss_type))

        self.other_loss_key = []
        for other_loss in args.other_loss.split('+'):
            self.other_loss_key.append(other_loss)

    def forward(self, output, gt):
        loss_log = self.get_init_loss_log()
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](output, gt)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                loss_log[l['type']] = effective_loss.item()

        loss_sum = sum(losses)
        loss_log['Total'] = loss_sum.item()

        return loss_sum, loss_log

    def get_init_loss_log(self):
        loss_log = {}
        loss_log['Total'] = 0.
        for l in self.loss:
            if l['function'] is not None:
                loss_log[l['type']] = 0.
        for ol in self.other_loss_key:
            loss_log[ol] = 0.
        return loss_log
