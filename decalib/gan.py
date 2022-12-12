import torch
import torch.nn as nn
from .utils.config import cfg

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

class GAN1D(nn.Module):
    def __init__(self, model_config, deca = None):
        super(GAN1D, self).__init__()
        self.generator = Generator1D(model_config)
        self.discriminator = Discriminator(model_config)
        self.deca = deca
        self.ldm_idx = model_config.dataset.lmk_idx
        self.vgg = VGGFace2Loss(pretrained_model=)
    
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
        opdict_in = self.deca.decode(codedict_in, rendering=True, vis_lmk=False, return_vis=False, use_detail=False)
        img_in = opdict_in['rendered_images']
        landmarks_in = opdict_in['verts'][:, self.ldm_idx, :]

        # real landmarks?
        if self.real_img_ldm:
            delta_landmarks = landmarks_in[:batch_size//2] - landmarks_in[batch_size//2:]
            delta_landmarks = torch.cat([delta_landmarks, -delta_landmarks], dim=0)
        else: delta_landmarks = batches['landmarks']

        # generate fake shape params
        shape_out = self.generator(delta_landmarks, shape_in) + shape_in
        # render fake images
        codedict_out = codedict_in
        codedict_out['shape'] = shape_out
        opdict_out = self.deca.decode(codedict_out, rendering=True, vis_lmk=False, return_vis=False, use_detail=False)
        img_out = opdict_out['rendered_images']
        landmarks_out = opdict_out['verts'][:, self.ldm_idx, :]
        
        # vgg features
        imgs = torch.cat([img_in, img_out], dim=0)
        F = self.vgg.forward_features(imgs)
        F_real, F_gen = F[:batch_size], F[batch_size:]

        return F_real, F_gen, landmarks_in, landmarks_out, delta_landmarks, imgs