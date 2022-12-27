import os
from loguru import logger
import torch
import torch.nn as nn

from decalib.utils.lossfunc import VGGFace2Loss
from .utils.config import cfg
from .utils import util

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Generator(nn.Module):
    '''
    Simple UNet Structures like:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    '''
    def __init__(self, model_config):
        super(Generator, self).__init__()
        in_channel = model_config.in_channel
        out_channel = model_config.out_channel
        encoder_nf = model_config.encoder_nf
        decoder_nf = model_config.decoder_nf[:len(encoder_nf)]
        final_nf = model_config.decoder_nf[len(encoder_nf):]

        ## Encoder
        self.pooling = [nn.MaxPool2d(2) for _ in range(len(encoder_nf))]
        self.encoder = nn.ModuleList()
        prev_nf = in_channel
        for i in range(len(encoder_nf)):
            blk = ConvBlock(prev_nf, encoder_nf[i], bias=False)
            self.encoder.append(blk)
            prev_nf = encoder_nf[i]
        ## Decoder
        self.upsampling = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) for _ in range(len(decoder_nf))]
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_nf)):
            blk = ConvBlock(prev_nf, decoder_nf[i], bias=False)
            self.decoder.append(blk)
            prev_nf = decoder_nf[i] + encoder_nf[-i-1]
        ## Final
        self.final = nn.ModuleList()
        for i in range(len(final_nf)):
            blk = ConvBlock(prev_nf, final_nf[i], bias=False)
            self.final.append(blk)
            prev_nf = final_nf[i]
        self.final.append(nn.Conv2d(prev_nf, out_channel, 1, 1, 0, 1, 1, False))

    def forward(self, x):
        x_hist = [x]
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            x_hist.append(x)
            x = self.pooling[i](x)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
            x = self.upsampling[i](x)
            x = torch.cat([x, x_hist.pop()], dim=1)
        for i in range(len(self.final)):
            x = self.final[i](x)
        return x

class Discriminator(nn.Module):
    '''
    Input: VGG features of size
    '''
    def __init__(self, model_config):
        super(Discriminator, self).__init__()
        in_channel = model_config.in_channel
        self.fc = nn.Linear(in_channel, 1)

    def forward(self, x):
        return self.fc(x)

class Generator1D(nn.Module):
    '''
    Input: 2 1D tensor
    Output: 1 1D tensor
    '''
    def __init__(self, model_config):
        super(Generator1D, self).__init__()
        ldm_len, shp_len = model_config.in_channel
        hidden_channel = model_config.hidden_channel
        out_channel = shp_len
        self.ldm_fc = nn.Sequential(
            nn.Linear(ldm_len, hidden_channel[0]//2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.shp_fc = nn.Sequential(
            nn.Linear(shp_len, hidden_channel[0]//2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.ModuleList()
        for i in range(len(hidden_channel)):
            if i == 0:
                continue
            else:
                self.fc.append(nn.Sequential(
                    nn.Linear(hidden_channel[i-1], hidden_channel[i]),
                    nn.LeakyReLU(0.2, inplace=True),
                ))
        self.out_fc = nn.Linear(hidden_channel[-1], out_channel)
    
    def forward(self, ldm, shp):
        ldm = self.ldm_fc(ldm)
        shp = self.shp_fc(shp)
        x = torch.cat([ldm, shp], dim=1)
        for i in range(len(self.fc)):
            x = self.fc[i](x)
        x = self.out_fc(x)
        return x

class GAN1D(nn.Module):
    def __init__(self, model_config, deca = None):
        super(GAN1D, self).__init__()
        self.vgg = VGGFace2Loss(pretrained_model=model_config.fr_model_path).eval()
        # freeze vgg
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.generator = Generator1D(model_config.generator)
        self.discriminator = Discriminator(model_config.discriminator)
        self.deca = deca
        self.ldm_idx = model_config.lmk_idx
        self.real_img_ldm = model_config.real_img_ldm
        self.cfg = model_config
    
    def load_chkp(self, path):
        model_chkp = path
        chkp = torch.load(model_chkp)
        cur_dict = self.model_dict()
        print(f'trained model found. load {model_chkp}')
        for k in chkp.keys():
            if k in cur_dict.keys():
                print('load module dict', k, 'from checkpoint')
                util.copy_state_dict(cur_dict[k], chkp[k])
    
    def model_dict(self):
        d =  dict(
            generator = self.generator.state_dict(),
            discriminator = self.discriminator.state_dict(),
        )
        d.update(self.deca.model_dict())
        return d
    
    def forward(self, batches):
        # self.deca.eval()
        # # freeze deca
        # for param in self.deca.parameters():
        #     param.requires_grad = False

        images = batches['images']
        batch_size = images.shape[0]
        # get camera, light, pose, expression, shape, texture
        codedict_in = self.deca.encode(images, use_detail=False)
        shape_in = codedict_in['shape']
        # real landmarks?
        if self.real_img_ldm:
            # Option 1: calculate landmarks in real settings
            if not self.cfg.normal_settings:
                opdict_in = self.deca.decode(codedict_in, rendering=True, vis_lmk=False, return_vis=False, use_detail=False)
                landmarks_in = opdict_in['verts'][:, self.ldm_idx, :]
                codedict_del = {}
                for k in codedict_in.keys():
                    if k not in ['shape', 'tex']:
                        codedict_del[k] = codedict_in[k]
                    else:
                        codedict_del[k] = codedict_in[k][list(range(batch_size//2, batch_size))+list(range(batch_size//2))]
                opdict_del = self.deca.decode(codedict_del, only_verts=True)
                landmarks_del = opdict_del['verts'][:, self.ldm_idx, :]
                delta_landmarks = landmarks_in - landmarks_del
            # Option 2: normalize settings
            else:
                codedict_norm = util.normalize_codedict(codedict_in.copy())
                opdict_norm = self.deca.decode(codedict_norm, rendering=True, vis_lmk=False, return_vis=False, use_detail=False)
                landmarks_in = opdict_norm['verts'][:, self.ldm_idx, :]
                landmarks_del = landmarks_in[list(range(batch_size//2, batch_size))+list(range(batch_size//2))]
                delta_landmarks = landmarks_in - landmarks_del
                opdict_in = opdict_norm
                codedict_in = codedict_norm
        # not supported yet
        else: delta_landmarks = batches['landmarks']

        img_in = opdict_in['rendered_images']
        # generate fake shape params
        shape_out = self.generator(delta_landmarks.view(batch_size,-1), \
            shape_in) + shape_in
        logdict = {
            'shape_out change max': (shape_out/shape_in-1).abs().max().item(),
            'shape_out change mean': (shape_out/shape_in-1).abs().mean().item(),
            'landmark_init_error': delta_landmarks.norm(dim=-1).mean().item(),
        }
        codedict_out = codedict_in.copy()
        codedict_out['shape'] = shape_out
        # use same texture as target images
        if self.cfg.use_target_tex:
            codedict_out['tex'] = codedict_in['tex'][list(range(batch_size//2, batch_size))+list(range(batch_size//2))]
        opdict_out = self.deca.decode(codedict_out, rendering=True, vis_lmk=False, return_vis=False, use_detail=False)
        img_out = opdict_out['rendered_images']
        landmarks_out = opdict_out['verts'][:, self.ldm_idx, :]
        logdict.update({
            'landmark_dist': ((landmarks_in-landmarks_out-delta_landmarks)).norm(dim=-1).mean().item()
        })
        # render fake images (everything but the tex and shape are as original images)
        # codedict_out = {}
        # for k in codedict_in.keys():
        #     if k not in ['shape', 'tex']:
        #         codedict_out[k] = codedict_in[k].view(2, batch_size//2, -1)[[1,0]].flatten(0,1)
        # codedict_out['tex'] = codedict_in['tex']
        
        # vgg features
        imgs = torch.cat([img_in, img_out], dim=0)
        # F = self.vgg.forward_features(imgs)
        # F_real, F_gen = F[:batch_size], F[batch_size:]
        F_real, F_gen = self.vgg.forward_features(img_in), self.vgg.forward_features(img_out)
        visdict = {
            'images': imgs[:batch_size],
            'gen_images': imgs[batch_size:],
            'verts': opdict_in['verts'],
            'gen_verts': opdict_out['verts'],
            'trans_verts': opdict_in['trans_verts'],
            'gen_trans_verts': opdict_out['trans_verts'],
            'lmk': opdict_in['trans_verts'][:, self.ldm_idx, :],
            'lmk_gen': opdict_out['trans_verts'][:, self.ldm_idx, :],
        }

        return F_real, F_gen, landmarks_in, landmarks_out, delta_landmarks, visdict, logdict