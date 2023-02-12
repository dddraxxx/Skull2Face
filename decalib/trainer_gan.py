from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

from tqdm import tqdm
from .utils import util
from loguru import logger

from .gan import GAN1D
from .dataset_gan import build_train, build_val

class Trainer:
    def __init__(self, model, config=None, device='cuda'):
        deca = model.to(device)
        self.cfg = config
        self.device = device
        self.batch_size = config.dataset.batch_size
        self.deca = deca
        # freeze deca
        self.gan = GAN1D(config.model, deca).train().to(device)
        self.optimizer()
        self.load_deca_checkpoint()
        if not self.cfg.train.get('resume_steps', False):
            self.global_step = 0
        if self.cfg.train.write_summary:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))
        for param in self.deca.parameters():
            param.requires_grad = False
        self.deca.eval().to(device)
    
    def prepare_data(self):
        self.train_dataset = build_train(self.cfg.dataset)
        self.val_dataset = build_val(self.cfg.dataset)
        logger.info('---- training data numbers: ', len(self.train_dataset))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                            num_workers=self.cfg.dataset.num_workers,
                            pin_memory=True,
                            drop_last=True)
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=8, shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=False)
        self.val_iter = iter(self.val_dataloader)

    def optimizer(self):
        self.optD = torch.optim.Adam(self.gan.discriminator.parameters(), lr=self.cfg.train.D_lr, betas=(0.5, 0.999))
        self.optG = torch.optim.Adam(self.gan.generator.parameters(), lr=self.cfg.train.G_lr, betas=(0.5, 0.999))

    def load_deca_checkpoint(self):
        model_dict = self.gan.model_dict()
        if self.cfg.train.resume and os.path.exists(os.path.join(self.cfg.output_dir, 'model.tar')):
            checkpoint = torch.load(os.path.join(self.cfg.output_dir, 'model.tar'))
            for key in model_dict.keys():
                if key in checkpoint.keys():
                    util.copy_state_dict(model_dict[key], checkpoint[key])
            util.copy_state_dict(self.optG.state_dict(), checkpoint['optG'])
            util.copy_state_dict(self.optD.state_dict(), checkpoint['optD'])
            self.global_step = checkpoint['global_step']
            logger.info(f"resume training from {os.path.join(self.cfg.output_dir, 'model.tar')}")
            logger.info(f"training start from step {self.global_step}")
        # load model weights only
        elif os.path.exists(self.cfg.pretrained_modelpath):
            checkpoint = torch.load(self.cfg.pretrained_modelpath)
            key = 'E_flame'
            util.copy_state_dict(model_dict[key], checkpoint[key])
            self.global_step = 0
        else:
            logger.info('model path not found, start training from scratch')
            self.global_step = 0
    
    def training_step(self, batches, step):
        logdict = {}
        optD, optG = self.optD, self.optG
        optD.zero_grad(set_to_none=True)
        for k in batches.keys():
            if k=='id': continue
            batches[k] = batches[k].to(self.device)
        F_real, F_gen, landmarks_in, landmarks_out, delta_landmarks, visdict, logdict = self.gan(batches)
        # discriminator loss
        logits_real, logits_gen = self.gan.discriminator(F_real), self.gan.discriminator(F_gen.detach())
        loss_d_real = F.binary_cross_entropy_with_logits(logits_real, F_real.new_ones(logits_real.shape))
        loss_d_gen = F.binary_cross_entropy_with_logits(logits_gen, F_gen.new_zeros(logits_gen.shape))
        loss_d = loss_d_real + loss_d_gen
        loss_d.backward()
        optD.step()
        loss_dict = {
            'loss_d': loss_d.item(),
            'loss_d_real': loss_d_real.item(),
            'loss_d_gen': loss_d_gen.item(),
        }

        # generator loss
        optG.zero_grad(set_to_none=True)
        logits_gen = self.gan.discriminator(F_gen)
        loss_g = F.binary_cross_entropy_with_logits(logits_gen, F_gen.new_ones(logits_gen.shape))
        # id loss enforce generated face to have similar features with initial face
        loss_id = self.gan.vgg._cos_metric(F_real, F_gen).mean()
        loss_lmk = ((landmarks_in-landmarks_out-delta_landmarks)*self.cfg.train.lmk_scale).pow(2).mean()
        logdict.update({
            'landmark_dist_scaled': (((landmarks_in-landmarks_out-delta_landmarks)*self.cfg.train.lmk_scale)**2).sum(-1).sqrt().mean().item()
        })
        loss = loss_g + loss_id + loss_lmk
        loss.backward()
        optG.step()
        loss_dict.update({
            'loss_g': loss_g.item(),
            'loss_id': loss_id.item(),
            'loss_lmk': loss_lmk.item(),
            'loss_g_all': loss.item(),
        })
        return loss_dict, visdict, logdict, visdict['lmk'], visdict['lmk_gen']
    
    def fit(self):
        self.prepare_data()

        iters_every_epoch = int(len(self.train_dataset)/self.batch_size)
        start_epoch = self.global_step//iters_every_epoch
        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            # for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.cfg.train.max_epochs}")):
            for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch+1}/{self.cfg.train.max_epochs}]"):
                if epoch*iters_every_epoch + step < self.global_step:
                    continue
                try:
                    batch = next(self.train_iter)
                except:
                    self.train_iter = iter(self.train_dataloader)
                    batch = next(self.train_iter)
                loss_dict, visdict, logdict, landmarks_in, landmarks_out = self.training_step(batch, step)

                # write summary
                if self.global_step % self.cfg.train.log_steps == 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                    for k, v in logdict.items():
                        loss_info = loss_info + f'{k}: {v:.4f}, '
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_log/'+k, v, global_step=self.global_step)
                    loss_info += '\n'
                    for k, v in loss_dict.items():
                        loss_info = loss_info + f'{k}: {v:.4f}, '
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/'+k, v, global_step=self.global_step)                    
                    logger.info(loss_info)

                if self.global_step % self.cfg.train.vis_steps == 0:
                    visind = list(range(4))
                    shape_images = self.deca.render.render_shape(visdict['verts'][visind], visdict['trans_verts'][visind])
                    gen_shape_images = self.deca.render.render_shape(visdict['gen_verts'][visind], visdict['gen_trans_verts'][visind])
                    visdict = {
                        'images': visdict['images'][visind],
                        'gen_images': visdict['gen_images'][visind],
                        'shape_images': shape_images,
                        'gen_shape_images': gen_shape_images,
                    }
                    lmkdict = {
                        'landmarks': landmarks_in[visind],
                        'gen_landmarks': landmarks_out[visind],
                    }
                    # put lmk on vis image
                    ldmd = {}
                    for k, v, l in zip(visdict.keys(), visdict.values(), lmkdict.values()):
                        ldmd['{}_ldm'.format(k)] = util.tensor_vis_landmarks(v.detach(), l.detach())
                    visdict.update(ldmd)
                    savepath = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir, f'{self.global_step:06}.jpg')
                    grid_image = util.visualize_grid(visdict, savepath, return_gird=True)
                    # import ipdb; ipdb.set_trace()                    
                    self.writer.add_image('train_images', (grid_image/255.).astype(np.float32).transpose(2,0,1), self.global_step)

                if self.global_step>0 and self.global_step % self.cfg.train.checkpoint_steps == 0:
                    model_dict = self.gan.model_dict()
                    model_dict['optG'] = self.optG.state_dict()
                    model_dict['optD'] = self.optD.state_dict()
                    model_dict['global_step'] = self.global_step
                    model_dict['batch_size'] = self.batch_size
                    torch.save(model_dict, os.path.join(self.cfg.output_dir, 'model' + '.tar'))   
                    # 
                    if self.global_step % self.cfg.train.checkpoint_steps*10 == 0:
                        os.makedirs(os.path.join(self.cfg.output_dir, 'models'), exist_ok=True)
                        torch.save(model_dict, os.path.join(self.cfg.output_dir, 'models', f'{self.global_step:08}.tar'))   

                # if self.global_step % self.cfg.train.val_steps == 0:
                #     self.validation_step()
                
                # if self.global_step % self.cfg.train.eval_steps == 0:
                #     self.evaluate()

                self.global_step += 1
                if self.global_step > self.cfg.train.max_steps:
                    break
