# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

from functools import reduce
import itertools
import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.gan import GAN1D
from decalib.datasets import datasets 
from decalib.utils import util
# from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

def main(cfg, args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    if args.savefolder is None:
        savefolder = os.path.join(args.inputpath, 'gan_results')
    else:
        savefolder = args.savefolder
    if args.preclear and os.path.exists(savefolder):
        if os.path.exists(savefolder+'_old'):
            os.system('rm -r ' + savefolder+'_old')
        os.system('mv ' + savefolder + ' ' + savefolder + '_old')
    device = args.device
    device = '{}:{}'.format(cfg.device, cfg.device_id)
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)

    # run DECA
    cfg.model.use_tex = args.useTex
    cfg.rasterizer_type = args.rasterizer_type
    cfg.model.extract_tex = args.extractTex
    deca = DECA(config = cfg, device=device).eval()
    gan = GAN1D(cfg.model, deca).eval().to(device)
    chkp_path = os.path.join(cfg.output_dir, 'model.tar')
    gan.load_chkp(chkp_path)
    # for i in range(len(testdata)):
    if cfg.test is None or cfg.test.get('pairs', None) is None:
        # test_pairs = [(0,j) for j in range(1, len(testdata))] + [(7,9)]
        # permute 0,1,4,7,9
        test_pairs = list(itertools.combinations([0,1,4,7,9], 2))
    else: test_pairs = cfg.test.pairs
    sum_dict = {}
    all_imgs = {}
    for i in set(reduce(lambda x,y: x+y, test_pairs)):
        all_imgs[testdata[i]['imagename']] = testdata[i]['image'][None]
    cv2.imwrite(os.path.join(savefolder, 'all_input.jpg'), deca.visualize(all_imgs, dim=2, title_key=True))
    print('save all input images to ', os.path.join(savefolder, 'all_input.jpg'))
    for i,j in tqdm(test_pairs):
        name = testdata[i]['imagename']+'__'+testdata[j]['imagename']
        image1 = testdata[i]['image'].to(device)[None,...]
        image2 = testdata[j]['image'].to(device)[None,...]
        with torch.no_grad():
            _, _, _, _, _, vis, logdict = gan({
                'images':torch.cat([image1, image2], dim=0)
            })
            print(logdict)
            print('lmk init error: ', (vis['lmk']-vis['lmk'].flip(0)).norm(dim=-1).mean())
            print('lmk error: ', (vis['lmk'].flip(0)-vis['lmk_gen']).norm(dim=-1).mean())
            vis['images'] = gan.deca.render.render_shape(vis['verts'], vis['trans_verts'])
            vis['gen_images'] = gan.deca.render.render_shape(vis['gen_verts'], vis['gen_trans_verts'])
            lmk_imgs = util.tensor_vis_landmarks(vis['images'], vis['lmk'])
            double_lmk_imgs = util.tensor_vis_landmarks(lmk_imgs, vis['lmk'].flip(0), color='r')
            lmk_gen_imgs = util.tensor_vis_landmarks(vis['gen_images'], vis['lmk_gen'], gt_landmarks=vis['lmk'].flip(0))
        visdict = {
            '{}_{}'.format(i,j): torch.stack([vis['images'][0], vis['gen_images'][0], vis['images'][1]], dim=0),
            '{}_{}'.format(j,i): torch.stack([vis['images'][1], vis['gen_images'][1], vis['images'][0]], dim=0),
            '{}_{}_lmk'.format(i,j): torch.stack([double_lmk_imgs[0], lmk_gen_imgs[0], lmk_imgs[1]], dim=0),
            '{}_{}_lmk'.format(j,i): torch.stack([double_lmk_imgs[1], lmk_gen_imgs[1], lmk_imgs[0]], dim=0),
        }
        sum_dict.update(visdict)
        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        if args.saveObj:
            # save original mesh
            deca.save_obj(os.path.join(savefolder, name, testdata[i]['imagename'] + '.obj'), dict(verts=vis['verts'][:1]), save_detail=False)
            deca.save_obj(os.path.join(savefolder, name, testdata[j]['imagename'] + '.obj'), dict(verts=vis['verts'][1:2]), save_detail=False)
            # save generated mesh
            deca.save_obj(os.path.join(savefolder, name, testdata[i]['imagename'] + '_gen.obj'), dict(verts=vis['gen_verts'][:1]), save_detail=False)
            deca.save_obj(os.path.join(savefolder, name, testdata[j]['imagename'] + '_gen.obj'), dict(verts=vis['gen_verts'][1:2]), save_detail=False)
        # if args.saveMat:
        #     opdict = util.dict_tensor2npy(opdict)
        #     savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
        if args.saveVis:
            cv2.imwrite(os.path.join(savefolder, name, 'vis.jpg'), deca.visualize({
                '{}'.format(testdata[i]['imagename']): testdata[i]['image'][None],
                '{}_shape'.format(testdata[i]['imagename']): vis['images'][0][None],
                '{}_gen_shape'.format(testdata[i]['imagename']): vis['gen_images'][0][None],
                '{}'.format(testdata[j]['imagename']): testdata[j]['image'][None],
                '{}_shape'.format(testdata[j]['imagename']): vis['images'][1][None],
                '{}_gen_shape'.format(testdata[j]['imagename']): vis['gen_images'][1][None],
            },dim=2, title_key=True))
            # save images
    for t in set(reduce(lambda x,y:x+y, test_pairs)):
        name = testdata[t]['imagename']
        # make dir
        os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        to_t_lmk = {}
        to_t = {}
        for k in sum_dict:
            if '_{}_lmk'.format(t) in k:
                to_t_lmk[k] = sum_dict[k] if k not in to_t_lmk else torch.cat([to_t_lmk[k], sum_dict[k]], dim=0)
            elif k.endswith('_{}'.format(t)):
                to_t[k] = sum_dict[k] if k not in to_t else torch.cat([to_t[k], sum_dict[k]], dim=0)
        cv2.imwrite(os.path.join(savefolder, name, '{}_lmk_vis.jpg'.format(testdata[t]['imagename'])), deca.visualize(to_t_lmk,dim=1))
        cv2.imwrite(os.path.join(savefolder, name, '{}_vis.jpg'.format(testdata[t]['imagename'])), deca.visualize(to_t,dim=1))
        # calculate diff between gen images
        diff_dct = {}
        gen_imgs = torch.stack(list(to_t.values()), dim=0)[:,1]
        if gen_imgs.shape[0] == 1:
            continue
        for i in range(gen_imgs.shape[0]):
            for j in range(i+1, gen_imgs.shape[0]):
                img1 = gen_imgs[i]
                img2 = gen_imgs[j]
                diff = (img1-img2).norm(dim=0)[None].repeat(3,1,1)
                diff_dct['{}_{}'.format(i,j)] = torch.stack([img1, diff, img2], dim=0)
        
        cv2.imwrite(os.path.join(savefolder, name, '{}_diff_vis.jpg'.format(testdata[t]['imagename'])), deca.visualize(diff_dct,dim=1))
    print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default=None, type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    parser.add_argument('-p', '--preclear', action='store_true')
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--mode', type=str, default = 'eval', help='deca mode')
    from decalib.utils.config import parse_args
    cfg, args = (parse_args(parser, True))
    main(cfg, args)