import logging
from collections import OrderedDict
from sys import settrace

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')
import model.losses as losses

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # set loss and load resume state
        # self.set_loss()
        self.mse = nn.MSELoss()
        self.criterion_char =losses.CharbonnierLoss()
        self.criterion_edge = losses.EdgeLoss()

        # self.set_new_noise_schedule(
            # opt['model']['beta_schedule']['train'], schedule_phase='train')
        
        if self.opt['phase'] == 'train':
            self.netG.train()
            # self.netd.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optim = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optim.zero_grad()
        LQ = self.data['LQ']
        type_index = self.data['index'][0]
        HQ = self.data['HQ']
        # import ipdb; ipdb.set_trace()

        Restore = self.netG(LQ, type_index)
        # need to average in multi-gpu
        label_img2 = F.interpolate(HQ, scale_factor=0.5, mode='bilinear')
        label_img4 = F.interpolate(HQ, scale_factor=0.25, mode='bilinear')
        # b, c, h, w = self.data['HQ'].shape
        # loss_d = loss_d.sum()/int(b*c*h*w)
        # loss_D = loss_D.sum()/int(b*c*h*w)
        loss_char_0 = self.criterion_char(Restore[0], label_img4)
        loss_char_1 = self.criterion_char(Restore[1], label_img2)
        loss_char_2 = self.criterion_char(Restore[2], HQ)
        loss_char = loss_char_0 + loss_char_1 + loss_char_2

        loss_edge_0 = self.criterion_edge(Restore[0], label_img4)
        loss_edge_1 = self.criterion_edge(Restore[1], label_img2)
        loss_edge_2 = self.criterion_edge(Restore[2], HQ)
        loss_edge = loss_edge_0 + loss_edge_1 + loss_edge_2

        label_fft1 = torch.rfft(label_img4, signal_ndim=2, normalized=False, onesided=False)
        pred_fft1 = torch.rfft(Restore[0], signal_ndim=2, normalized=False, onesided=False)
        label_fft2 = torch.rfft(label_img2, signal_ndim=2, normalized=False, onesided=False)
        pred_fft2 = torch.rfft(Restore[1], signal_ndim=2, normalized=False, onesided=False)
        label_fft3 = torch.rfft(HQ, signal_ndim=2, normalized=False, onesided=False)
        pred_fft3 = torch.rfft(Restore[2], signal_ndim=2, normalized=False, onesided=False)

        f1 = self.mse(pred_fft1, label_fft1)
        f2 = self.mse(pred_fft2, label_fft2)
        f3 = self.mse(pred_fft3, label_fft3)
        loss_fft = f1+f2+f3

        loss_tot = loss_char + 0.05*loss_edge + 0.1 * loss_fft
        loss_tot.backward()
        self.optim.step()

        # set log
        self.log_dict['loss_char'] = loss_char.item()
        self.log_dict['loss_edge'] = loss_edge.item()
        self.log_dict['loss_fft'] = loss_fft.item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            Restore = self.netG(self.data['LQ'], self.data['index'][0])
            self.Restore = Restore[-1]
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.Restore = self.netG.module.sample(batch_size, continous)[-1]
            else:
                self.Restore = self.netG.sample(batch_size, continous)[-1]
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.Restore.detach().float().cpu()
        else:
            out_dict['Restore'] = self.Restore.detach().float().cpu()
            out_dict['HQ'] = self.data['HQ'].detach().float().cpu()
            out_dict['LQ'] = self.data['LQ'].detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format('latest', 'latest'))
            # self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format('latest', 'latest'))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optim.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                logger.info(
                'Loading optimizer for G [{:s}] ...'.format(load_path))
                self.optim.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
