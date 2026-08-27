import logging
import random
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.Quantization import Quantization
from .modules.common import DWT,IWT

logger = logging.getLogger('base')
dwt=DWT()
iwt=IWT()


class Model_VSN(BaseModel):
    def __init__(self, opt):
        super(Model_VSN, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.gop = opt['gop']
        train_opt = opt['train']
        test_opt = opt.get('test', {})
        if test_opt is None:
            test_opt = {}
        self.opt = opt
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.opt_net = opt['network_G']
        self.center = self.gop // 2
        self.num_video = opt['num_video']
        self.active_num_video = self.num_video
        self.current_secret_indices = list(range(self.num_video))
        self.scalable_training = bool(opt.get('scalable_training', False))
        candidates = opt.get('active_num_video_candidates', None)
        if candidates is None:
            candidates = list(range(1, self.num_video + 1))
        self.active_num_video_candidates = sorted(
            [int(v) for v in candidates if 1 <= int(v) <= self.num_video]
        )
        if not self.active_num_video_candidates:
            self.active_num_video_candidates = [self.num_video]
        self.scalable_random_subset = bool(opt.get('scalable_random_subset', True))
        self.train_frame_offsets = opt.get('train_frame_offsets', [0])
        if self.train_frame_offsets is None:
            self.train_frame_offsets = [0]
        self.single_secret_only = bool(opt.get('single_secret_only', False))
        if self.single_secret_only and self.num_video != 1:
            raise ValueError('single_secret_only is enabled, but num_video is not 1.')
        self.idxx = 0

        self.netG = networks.define_G_v2(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()
        self.use_amp = bool(self.train_opt.get('use_amp', False) and torch.cuda.is_available()) if self.is_train else False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if self.is_train:
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])
            self.Reconstruction_center = ReconstructionLoss(losstype="center")

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  
        self.real_H = data['GT'].to(self.device)

    def init_hidden_state(self, z):
        b, c, h, w = z.shape
        h_t = []
        c_t = []
        device = z.device
        for _ in range(self.opt_net['block_num_rbm']):
            h_t.append(torch.zeros([b, c, h, w], device=device))
            c_t.append(torch.zeros([b, c, h, w], device=device))
        memory = torch.zeros([b, c, h, w], device=device)

        return h_t, c_t, memory

    def loss_forward(self, out, y):
        if self.opt['model'] == 'LSTM-VRN':
            l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
            return l_forw_fit
        elif self.opt['model'] == 'MIMO-VRN-h':
            l_forw_fit = 0
            for i in range(out.shape[1]):
                l_forw_fit += self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out[:, i], y[:, i])
            return l_forw_fit

    def loss_back_rec(self, out, x):
        if self.opt['model'] == 'LSTM-VRN':
            l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, x)
            return l_back_rec
        elif self.opt['model'] == 'MIMO-VRN-h':
            l_back_rec = 0
            for i in range(x.shape[1]):
                l_back_rec += self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out[:, i], x[:, i])
            return l_back_rec
    
    def loss_back_rec_mul(self, out, x):
        out = torch.chunk(out,self.num_video,dim=1)
        out = [outi.squeeze(1) for outi in out]
        x = torch.chunk(x,self.num_video,dim=1)
        x = [xi.squeeze(1) for xi in x]
        l_back_rec = 0
        for i in range(len(x)):
            for j in range(x[i].shape[1]):
                l_back_rec += self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out[i][:, j], x[i][:, j])
        return l_back_rec

    def loss_center(self, out, x):
        # x.shape: (b, t, c, h, w)
        b, t = x.shape[:2]
        l_center = 0
        for i in range(b):
            mse_s = self.Reconstruction_center(out[i], x[i])
            mse_mean = torch.mean(mse_s)
            for j in range(t):
                l_center += torch.sqrt((mse_s[j] - mse_mean.detach()) ** 2 + 1e-18)
        l_center = self.train_opt['lambda_center'] * l_center / b

        return l_center

    def _valid_offsets(self, frame_count, intval, offsets):
        center = frame_count // 2
        valid_offsets = []
        for offset in offsets:
            offset = int(offset)
            start = center - intval + offset
            end = center + intval + 1 + offset
            if start >= 0 and end <= frame_count:
                valid_offsets.append(offset)
        if not valid_offsets:
            valid_offsets = [0]
        return valid_offsets

    def _choose_train_offset(self, frame_count, intval):
        valid_offsets = self._valid_offsets(frame_count, intval, self.train_frame_offsets)
        return random.choice(valid_offsets)

    def _choose_active_num_video(self, is_train=True):
        if not self.scalable_training:
            return self.num_video
        if is_train:
            return random.choice(self.active_num_video_candidates)
        active_num = self.test_opt.get('active_num_video', self.num_video)
        if active_num is None:
            active_num = self.num_video
        active_num = int(active_num)
        active_num = max(1, min(active_num, self.num_video))
        return active_num

    def _prepare_active_secret(self, secret_all, active_num, random_subset=False):
        n_total = secret_all.shape[1]
        if active_num >= n_total:
            idx = list(range(n_total))
            return secret_all, idx
        idx = list(range(n_total))
        if random_subset:
            random.shuffle(idx)
        idx = sorted(idx[:active_num])
        idx_tensor = torch.tensor(idx, device=secret_all.device, dtype=torch.long)
        return torch.index_select(secret_all, 1, idx_tensor), idx

    def optimize_parameters(self, current_step):
        self.optimizer_G.zero_grad(set_to_none=True)

        b, n_total, t, c, h, w = self.ref_L.shape
        intval = self.gop // 2
        center = t // 2
        active_n = self._choose_active_num_video(is_train=True)
        self.active_num_video = active_n
        secret_all, secret_indices = self._prepare_active_secret(
            self.ref_L, active_n, random_subset=self.scalable_random_subset
        )
        self.current_secret_indices = secret_indices
        train_offset = self._choose_train_offset(t, intval)

        host = self.real_H[:, center - intval + train_offset:center + intval + 1 + train_offset]
        secret_window = secret_all[:, :, center - intval + train_offset:center + intval + 1 + train_offset]

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            secret_dwt = [dwt(secret_window[:, i].reshape(b, -1, h, w)) for i in range(active_n)]
            output, _ = self.netG(x=dwt(host.reshape(b, -1, h, w)), x_h=secret_dwt)
            output = iwt(output)

            container = output[:, :3 * self.gop, :, :].reshape(-1, self.gop, 3, h, w)[:, self.gop // 2]
            l_forw_fit = self.loss_forward(container.unsqueeze(1), host[:, self.gop // 2].unsqueeze(1))

            y = self.Quantization(
                output[:, :3 * self.gop, :, :].view(-1, self.gop, 3, h, w)[:, self.gop // 2]
                .unsqueeze(1).repeat(1, self.gop, 1, 1, 1).reshape(b, -1, h, w)
            )
            out_x, out_x_h, _ = self.netG(x=dwt(y), rev=True, num_active=active_n)
            out_x = iwt(out_x)
            out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]

            l_back_rec = self.loss_back_rec(
                out_x.reshape(-1, self.gop, 3, h, w)[:, self.gop // 2].unsqueeze(1),
                host[:, self.gop // 2].unsqueeze(1)
            )

            out_x_h = torch.stack(out_x_h, dim=1)
            out_x_h = out_x_h.reshape(-1, active_n, self.gop, 3, h, w)
            secret_center = secret_window[:, :, self.gop // 2]

            l_secret_rec = 0
            for i in range(active_n):
                l_secret_rec += self.loss_back_rec(
                    out_x_h[:, i, self.gop // 2].unsqueeze(1),
                    secret_center[:, i].unsqueeze(1)
                )

            secret_weight = float(self.train_opt.get('lambda_secret_weight', 4.0))
            forward_weight = float(self.train_opt.get('lambda_forward_weight', 1.0))
            host_weight = float(self.train_opt.get('lambda_host_weight', 1.0))
            loss = forward_weight * l_forw_fit + host_weight * l_back_rec + secret_weight * l_secret_rec

        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.train_opt['gradient_clipping']:
                self.scaler.unscale_(self.optimizer_G)
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
            self.scaler.step(self.optimizer_G)
            self.scaler.update()
        else:
            loss.backward()
            if self.train_opt['gradient_clipping']:
                nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])
            self.optimizer_G.step()

        self.host = host
        self.secret = secret_window
        self.output = output

        self.log_dict['l_back_rec'] = l_back_rec.item()
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_secret_rec'] = l_secret_rec.item()
        self.log_dict['active_n'] = float(active_n)

    def _get_test_offsets(self, frame_count, intval):
        default_offsets = self.opt.get('test_frame_offsets', [-1, 0, 1])
        if default_offsets is None:
            default_offsets = [-1, 0, 1]
        return self._valid_offsets(frame_count, intval, default_offsets)

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            forw_L = []
            forw_L_h = []
            fake_H = []
            fake_H_h = []
            pred_z = []
            b, t, c, h, w = self.real_H.shape
            center = t // 2
            intval = self.gop // 2
            ids = self._get_test_offsets(t, intval)
            b, n_total, t, c, h, w = self.ref_L.shape
            active_n = self._choose_active_num_video(is_train=False)
            self.active_num_video = active_n
            secret_all, secret_indices = self._prepare_active_secret(
                self.ref_L, active_n, random_subset=False
            )
            self.current_secret_indices = secret_indices
            for id in ids:
                # forward downscaling
                self.host = self.real_H[:, center - intval+id:center + intval + 1+id]
                secret_window = secret_all[:, :, center - intval + id:center + intval + 1 + id]
                secret_dwt = [dwt(secret_window[:, i].reshape(b, -1, h, w)) for i in range(active_n)]
                self.output, out_h = self.netG(x=dwt(self.host.reshape(b, -1, h, w)), x_h=secret_dwt)
                self.output = iwt(self.output)
                out_lrs = self.output[:, :3 * self.gop, :, :].reshape(-1, self.gop, 3, h, w)

                # backward upscaling
                y = self.Quantization(self.output[:, :3 * self.gop, :, :].view(-1, self.gop, 3, h, w)[:,self.gop//2].unsqueeze(1).repeat(1,self.gop,1,1,1).reshape(b, -1, h, w))
                out_x, out_x_h, out_z = self.netG(x=dwt(y), rev=True, num_active=active_n)
                out_x = iwt(out_x)
                out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]
                out_x = out_x.reshape(-1, self.gop, 3, h, w)
                out_x_h = torch.stack(out_x_h, dim=1)
                out_x_h = out_x_h.reshape(-1, active_n, self.gop, 3, h, w)
                forw_L.append(out_lrs[:, self.gop//2])
                fake_H.append(out_x[:, self.gop//2])
                fake_H_h.append(out_x_h[:,:, self.gop//2])

            self.secret = secret_all[:, :, center - intval:center + intval + 1]

        self.fake_H = torch.clamp(torch.stack(fake_H, dim=1),0,1)
        self.fake_H_h = torch.clamp(torch.stack(fake_H_h, dim=2),0,1)
        self.forw_L = torch.clamp(torch.stack(forw_L, dim=1),0,1)
        self.netG.train()
        
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        b, n, t, c, h, w = self.ref_L.shape
        center = t // 2
        intval = self.gop // 2
        active_n = int(getattr(self, 'active_num_video', n))
        active_n = max(1, min(active_n, n))
        active_indices = getattr(self, 'current_secret_indices', list(range(active_n)))
        if len(active_indices) < active_n:
            active_indices = list(range(active_n))

        idx_tensor = torch.tensor(active_indices[:active_n], device=self.ref_L.device, dtype=torch.long)
        ref_active = torch.index_select(self.ref_L, 1, idx_tensor)
        out_dict = OrderedDict()
        LR_ref = ref_active[:, :, center - intval:center + intval + 1].detach()[0].float().cpu()
        LR_ref = torch.chunk(LR_ref, active_n, dim=0)
        out_dict['LR_ref'] = [video.squeeze(0) for video in LR_ref]
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        SR_h = self.fake_H_h.detach()[0].float().cpu()
        SR_h = torch.chunk(SR_h, active_n, dim=0)
        out_dict['SR_h'] = [video.squeeze(0) for video in SR_h]
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H[:, center - intval:center + intval + 1].detach()[0].float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
    
    def load_test(self,load_path_G):
        self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
