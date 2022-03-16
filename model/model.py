import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import numpy as np
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
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
        loss_d, loss_D = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HQ'].shape
        loss_d = loss_d.sum()/int(b*c*h*w)
        loss_D = loss_D.sum()/int(b*c*h*w)
        loss_tot = loss_d + loss_D
        loss_tot.backward()
        self.optim.step()

        # set log
        self.log_dict['loss_d'] = loss_d.item()
        self.log_dict['loss_D'] = loss_D.item()

    def test(self, continous=False):
        self.netG.eval()
        if self.opt['sample']['sample_type'] == "generalized":
            tot_timestep = self.opt['sample']['n_timestep']
            if self.opt['sample']['skip_type'] == 'uniform':
                skip = tot_timestep // self.opt['sample']['sample_step']
                seq = range(0, tot_timestep, skip)
            elif self.opt['sample']['sample_type'] == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.tot_timestep * 0.8), self.opt['sample']['sample_step']
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)] 
            with torch.no_grad():
                if isinstance(self.netG, nn.DataParallel):
                    self.Restore = self.netG.module.skip_restore(
                        self.data['LQ'], seq, continous)
                else:
                    self.Restore = self.netG.skip_restore(
                        self.data['LQ'], seq, continous)
        elif self.opt['sample']['sample_type'] == "ddpm":
            with torch.no_grad():
                if isinstance(self.netG, nn.DataParallel):
                    self.Restore = self.netG.module.restore(
                        self.data['LQ'], continous)
                else:
                    self.Restore = self.netG.restore(
                        self.data['LQ'], continous)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.Restore = self.netG.module.sample(batch_size, continous)
            else:
                self.Restore = self.netG.sample(batch_size, continous)
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
            # out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HQ'] = self.data['HQ'].detach().float().cpu()
            # if need_LR and 'LR' in self.data:
            #     out_dict['LR'] = self.data['LR'].detach().float().cpu()
            # else:
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
