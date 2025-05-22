'''
Inference code for VisTR
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import math
import os
import sys
#sys.path.append(os.path.join(os.path.dirname(__file__)))
from typing import Iterable
from PIL import Image
import torch
import torch.nn.functional as F
import datasets2.transforms as T
from util.util_vec import *
import numpy as np
import glob
import time
import datetime
from pathlib import Path
import util.misc as utils
import csv , time
from models.vctran_colorvid import build_model
from models.transformer_inter import TransformerInternLayer
from models.FID import FID_utils,LPIP_utils 
from models.warping import WarpingLayer
from util.flowlib import read_flow
import torch.backends.cudnn as cudnn
from tqdm import tqdm

# subdir
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # Model parameters
    parser.add_argument('--pretrained_weights', type=str, default="r101_pretrained.pth",
                        help="Path to the pretrained model.")
    parser.add_argument('--model_path', type=str, 
                        default="./stage2/checkpoints/checkpoint_encoder_finetune02010000.pth")
    parser.add_argument('--decoder_path', type=str, 
                        default="./stage2/checkpoints/checkpoint_decoder_finetune02010000.pth")
    # * Backbone
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--lr_backbone', default=0, type=int)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--test_path', type=str,default='./dataset/temp/test_input/difficult_input')  #/dataset/videvo/test/imgs /dataset/DAVIS/Test/1/imgs

    parser.add_argument('--test_output_path', type=str,default='./stage2_difficult_results') 
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    parser.add_argument("--img_size", type=int, default=[480 , 848] )  
    parser.add_argument("--nhead_warp", type=int, default=2 ) 
    parser.add_argument('--num_frames', default=4, type=int,
                        help="Number of frames")
    parser.add_argument('--scale_size', default=[216,384], type=float)                    

    parser.add_argument('--ref_path', default='./stage1_difficult_results',)  #./stage1_test_results

    return parser


    '''
    args.num_frames   test img length per clip, note that it doesn't contain ref img
    args.test_path    test img dir e.g. Davis/Val/
    args.test_output_path  remember to define your output path for per specific trained model
    args.out_csv_name     name of output csv file
    args.decoder_path     path of decoder
    args.model_path       path of encoder
    args.img_size         test img size
    args.scale_factor     scale imgs to accelerate inference
    args.mode             linear,parallel for decoder ; testonly when not do testing ; note that interlayers closed  in linear mode; nowe when do not caculate warp error.
    '''


@torch.no_grad()
def main_testonly_parallel(args):
    print("mode testonly!")

    # Loading Modls and Weights
    device = torch.device(args.device)
    model, decoder  = build_model(args)
    interlayer_save = TransformerInternLayer(384,384,4)
    model.to(device)
    decoder.to(device)
    interlayer_save.to(device)
    checkpoint_vctran = torch.load(args.model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint_vctran['model'],strict=True)
    interlayer_save.load_state_dict(checkpoint_vctran['inter_save'],strict=True)
    print("loaded encoder weight")
    checkpoint_decoder = torch.load(args.decoder_path, map_location='cpu', weights_only=False)
    decoder.load_state_dict(checkpoint_decoder['model'],strict=True)
    print("loaded decoder weight")

    # Transition to eval mode
    model.eval()
    decoder.eval()
    interlayer_save.eval()

    # Freeze parameters in models and decoder
    for p in model.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False
    for p in interlayer_save.parameters():
        p.requires_grad = False
    

    n_parameters_model = sum(p.numel() for n,p in model.named_parameters())
    n_parameters_decoder = sum(p.numel() for n,p in decoder.named_parameters())        #计算参数量的方法
    n_parameters_inter = sum(p.numel() for n,p in interlayer_save.named_parameters()) 
    n_parameters_all = n_parameters_model+n_parameters_decoder+n_parameters_inter
    print('number of params:',n_parameters_all)

    #-----------------------------列出待处理视频子文件夹--------------------------------#
    test_num_frames = args.num_frames
    subdirs = sorted(os.listdir(args.test_path))
    # print 待处理目录
    print("subdirs:", subdirs)

    for index, subdir in enumerate(tqdm(subdirs)):
        print(f"Now processing subdir {subdir}...")
        path = os.path.join(args.test_path, subdir)
        imgs = glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.jpg'))
        print(f"Found {len(imgs)} images in {path}")
    print("\n")

    if args.ref_path:
        reference_imgs = sorted(glob.glob(os.path.join(args.ref_path,"*.JPEG"))+glob.glob(os.path.join(args.ref_path,"*.png")))
        print("reference_imgs:", reference_imgs)
        print("len(reference_imgs):", len(reference_imgs))
        print("args.ref_path:", args.ref_path)
    print("\n")


    # —— 【新增】 跨段比对时记住上一个 segment 的 ref —— #
    prev_ref_tensor = None


    #对每个子段（subdir）的处理流程
    for index,subdir in enumerate(tqdm(subdirs)):
        torch.cuda.empty_cache()

        #在新的subdir开始时，加入我们的操作。
        print("Now processing subdir", subdir)

        # path = os.path.join(args.test_path, subdir,"2-1")
        path = os.path.join(args.test_path, subdir)
        # print("precessing:",path)

        #imgs = sorted()
        imgs = glob.glob(os.path.join(path, '*.png'))+glob.glob(os.path.join(path, '*.jpg'))
        imgs.sort(key=lambda f: int("".join(filter(str.isdigit, f) or -1)))
        #print(imgs)

        Clip = []
        for i in range(0,len(imgs),1):
            img = Image.open(imgs[i]).convert('RGB')
            Clip.append(transform(img).unsqueeze(0))
        Clip = torch.cat(Clip,dim=0)
        Clip.requires_grad =False

        # —— 4.2) 本段的 reference：第 index 张预生成图 —— #
        if args.ref_path:
            ref = reference_imgs[index]
            ref = Image.open(ref).convert('RGB')
            ref = transform(ref).unsqueeze(0).cuda()
            #print(ref.shape)
        else:
            ref = Clip[0:1,:,:,:]  # fallback


        #---------------- —— 【插入点】 跨段 ref 比对 —— --------------------#
        if prev_ref_tensor is not None:
            # 计算两张 ref 的直方图 JS divergence
            jsd = compute_hist_jsd(prev_ref_tensor, ref)
            if jsd > args.jsd_threshold:
                # 如果差异太大，执行直方图匹配
                ref = histogram_match(prev_ref_tensor, ref)
        # 更新 prev_ref_tensor
        prev_ref_tensor = ref.detach().clone()
        #---------------- —— 【插入点】 跨段 ref 比对 —— --------------------#


        tail_flag = 0  
        # 标记本段末尾窗口是否不足 test_num_frames，若是则动态调整模型并在后面恢复
        # “跨-ref”逻辑一般不修改这行

        # 以 (test_num_frames-1) 为步长，依次取视频段的滑动窗口片段
        for j in range(0, len(Clip), test_num_frames - 1):
            # 取出滑窗内除 reference 之外的帧，[0:test_num_frames-1]
            clip = Clip[j:j + test_num_frames - 1, :, :, :].cuda()
            # GT = GGT[j:j+test_num_frames-1,:,:,:]  # 如需评估可打开

            # 把本段 reference 前置拼接到 clip 首位，构成 B=test_num_frames 大小的输入
            clip_large = torch.cat([ref, clip], dim=0)
            # 下采样到 transformer 输入尺寸，加速推理
            clip = F.interpolate(clip_large, size=args.scale_size, mode="bilinear")

            corr_num_frames = clip.shape[0]
            if corr_num_frames < test_num_frames:
                # 如果末尾窗口长度不足，则临时修改模型参数以适配更短序列
                model.num_frames = corr_num_frames
                model.backbone[1].frames = corr_num_frames
                args.num_frames = corr_num_frames
                tail_flag = 1  
                # “跨-ref”逻辑一般不改这里

            # 分离 L 通道（灰度）和 reference 的 ab 通道
            clip_l = clip[:, 0:1, :, :]        # shape [B,1,H,W]
            clip_ref_ab = clip[0:1, 1:3, :, :] # shape [1,2,H,W]

            # 除 reference 外，其余灰度帧先粗略转回 RGB，供模型输入
            clip_rgb_from_gray = gray2rgb_batch(clip_l[1:]).to(device)  
            # “跨-ref”不改这里

            # 将真正的 reference Lab 拼成 RGB
            I_reference_rgb = tensor_lab2rgb(
                torch.cat((uncenter_l(clip_l[0:1, :, :, :]), clip_ref_ab), dim=1)
            )
            # 拼接成完整 B 帧项目入网络
            clip_rgb_from_gray = torch.cat([I_reference_rgb, clip_rgb_from_gray], dim=0)

            # —— 主推理流程 —— #
            with torch.no_grad():
                if j == 0:
                    # 第一个窗口：不带历史 features
                    out, out_trans, features, pos = model(clip_rgb_from_gray)
                else:
                    # 其余窗口：携带上个窗口残余 features_inter
                    out, out_trans, features, pos = model(
                        clip_rgb_from_gray, features_inter
                    )
                # 解码：融合 warp 与 color
                out, warped_result, _ = decoder(
                    out, out_trans,
                    clip_ref_ab.unsqueeze(0),  # [1,2,H,W]
                    features,
                    clip_l.unsqueeze(0),       # [1,1,H,W]
                    pos=pos,
                    temperature_warp=1e-10
                )
                # 更新跨窗口记忆
                if j == 0:
                    features_inter = out_trans
                else:
                    features_inter = interlayer_save(features_inter, out_trans)
            # 清理显存
            del features, warped_result

            # 去掉 batch 维度
            out_ab = out.squeeze(0)  
            # 将 reference+clip_l 送回 CPU
            clip_l_large = clip_large[:, 0:1, :, :].data.cpu()
            # 恢复到最终输出分辨率
            out_ab = F.interpolate(
                out_ab, size=args.img_size, mode="bilinear"
            ).data.cpu()

            # 若之前修改过 num_frames，恢复原值
            if tail_flag:
                model.num_frames = test_num_frames
                model.backbone[1].frames = test_num_frames
                args.num_frames = test_num_frames
                tail_flag = 0

            # 将 windows 内除 reference 外的每一帧结果写出
            for i in range(corr_num_frames - 1):
                clip_l_corr = clip_l_large[i + 1 : i + 2, :, :, :]
                out_ab_corr  = out_ab[i + 1 : i + 2, :, :, :]
                outputs_rgb  = batch_lab2rgb_transpose_mc(clip_l_corr, out_ab_corr)
                output_path  = os.path.join(args.test_output_path, subdir)
                mkdir_if_not(output_path)
                save_frames(
                    outputs_rgb, output_path,
                    image_name="f%03d.png" % (j + i + 1)
                )
        print("done!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    cudnn.benchmark = True
    transform = T.Compose([
        T.CenterPad(args.img_size), T.RGB2Lab(), T.ToTensor(), T.Normalize()
        ])
    transform_GT = T.Compose([
        T.CenterPad(args.img_size),#T.ToTensor()
        ])
    transform2 = T.Compose(
        [T.CenterPad_vec(args.img_size), T.ToTensor()]
    )
    transform3 = T.Compose(
        [T.CenterPad(args.img_size), T.ToTensor()]
    )
    print("start stage2!")
    main_testonly_parallel(args)
    print("done stage2!")
    