import torch
import torch.nn as nn
import torch.nn.functional as F
from gan import GAN1D

class Trainer:
    def __init__(self, model, config=None, device='cuda'):
        deca = model.to(device)
        self.deca = deca
        self.deca.eval()
        # freeze deca
        for param in self.deca.parameters():
            param.requires_grad = False
        self.gan = GAN1D(config.model, deca)
    
    def training_step(self, batches):
        F_real, F_gen, landmarks_in, landmarks_out, delta_landmarks, imgs = self.gan(batches)
        optD, optG = self.optD, self.optG
        # discriminator loss
        optD.zero_grad(set_to_none=True)
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
        loss_id = self.gan.vgg._cos_metric(F_real, F_gen)
        loss_lmk = (landmarks_in-landmarks_out-delta_landmarks).pow(2).mean()
        loss = loss_g + loss_id + loss_lmk
        loss.backward()
        optG.step()
        loss_dict.update({
            'loss_g': loss_g.item(),
            'loss_id': loss_id.item(),
            'loss_lmk': loss_lmk.item(),
        })
        return loss_dict, imgs
