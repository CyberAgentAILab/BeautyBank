import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from argparse import Namespace
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm
from PIL import Image
from util import data_sampler, requires_grad, accumulate, sample_data, d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize, mixing_noise

from model.BeautyBank import BeautyBank
from model.stylegan.model import Discriminator
from model.encoder.psp import pSp
from model.encoder.criteria import id_loss
from model.vgg import VGG19
import model.contextual_loss.functional as FCX

try:
    import wandb

except ImportError:
    wandb = None

from model.stylegan.dataset import MultiResolutionDataset
from model.stylegan.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from model.stylegan.non_leaking import augment, AdaptiveAugment
from model.stylegan.model import Discriminator


class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Fine-tune BeautyBank")
        self.parser.add_argument("--style", type=str, default="makeup", help="style type")
        self.parser.add_argument("--iter", type=int, default=1000, help="total training iterations")
        self.parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
        self.parser.add_argument("--n_sample", type=int, default=9, help="number of the samples generated during training")
        self.parser.add_argument("--size", type=int, default=1024, help="image sizes for the model")
        self.parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        self.parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
        self.parser.add_argument("--path_batch_shrink", type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory consumption)")
        self.parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
        self.parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization")
        self.parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
        self.parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
        self.parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
        self.parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
        self.parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
        self.parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        self.parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
        self.parser.add_argument("--augment_p", type=float, default=0, help="probability of applying augmentation. 0 = use adaptive augmentation")
        self.parser.add_argument("--ada_target", type=float, default=0.6, help="target augmentation probability for adaptive augmentation")
        self.parser.add_argument("--ada_length", type=int, default=500 * 1000, help="target duraing to reach augmentation probability for adaptive augmentation")
        self.parser.add_argument("--ada_every", type=int, default=256, help="probability update interval of the adaptive augmentation")
        self.parser.add_argument("--start_iter", type=int, default=0, help="start iteration")
        self.parser.add_argument("--save_every", type=int, default=100, help="interval of saving a checkpoint")
        self.parser.add_argument("--save_begin", type=int, default=1000, help="when to start saving a checkpoint")
        self.parser.add_argument("--subspace_freq", type=int, default=2, help="how often to use paired data")
        self.parser.add_argument("--style_loss", type=float, default=0.25, help="the weight of feature matching loss")
        self.parser.add_argument("--CX_loss", type=float, default=0.25, help="the weight of contextual loss")
        self.parser.add_argument("--perc_loss", type=float, default=1.0, help="the weight of perceptual loss")
        self.parser.add_argument("--id_loss", type=float, default=1.0, help="the weight of identity loss")
        self.parser.add_argument("--L2_reg_loss", type=float, default=0.015, help="the weight of identity loss")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path to the saved models")    
        self.parser.add_argument("--encoder_path", type=str, default=None, help="path to the encoder model")
        self.parser.add_argument("--image_path", type=str, default=None, help="path to the image dataset")
        self.parser.add_argument("--identity_path", type=str, default=None, help="path to the identity model")
        self.parser.add_argument("--lmdb_path", type=str, default=None, help="path to the lmdb dataset")
        self.parser.add_argument("--makeup_path", type=str, default=None, help="path to the makeup codes")
        self.parser.add_argument("--bareface_path", type=str, default=None, help="path to the bareface codes")
        self.parser.add_argument("--model_name", type=str, default='generator', help="name of saved model")

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.encoder_path is None:
            self.opt.encoder_path = os.path.join(self.opt.model_path, 'encoder.pt')
        if self.opt.ckpt is None:
            self.opt.ckpt = os.path.join(self.opt.model_path, 'generator-pretrain.pt')   
        if self.opt.identity_path is None:
            self.opt.identity_path = os.path.join(self.opt.model_path, 'model_ir_se50.pth')     
        if self.opt.image_path is None:
            self.opt.image_path = os.path.join('./data', self.opt.style, 'images/train/')   
        if self.opt.lmdb_path is None:
            self.opt.lmdb_path = os.path.join('./data', self.opt.style, 'lmdb/')              
        if self.opt.makeup_path is None:
            self.opt.makeup_path = os.path.join(self.opt.model_path, self.opt.style, 'makeup_code.npy')    
        if self.opt.bareface_path is None:
            self.opt.bareface_path = os.path.join(self.opt.model_path, self.opt.style, 'bareface_code.npy')   
        args = vars(self.opt)
        if self.opt.local_rank == 0:
            print('Load options')
            for name, value in sorted(args.items()):
                print('%s: %s' % (str(name), str(value)))
        return self.opt                


def get_paired_data(barefaces, Simgs, makeups, subspace_std=0.1, batch_size=4, random_ind=8):
    ind = np.random.randint(0, barefaces.size(0), size=batch_size)
    Simg = Simgs[ind]
    bareface = barefaces[ind]
    if random_ind < 18:
        # eliminate the color consistency between the content image and the target image
        bareface[:,random_ind:18] = torch.normal(bareface[:,random_ind:18] * 0, 1.0)
    # add small jitters to extrinsic style to prevent overfiting
    makeup = torch.normal(makeups[ind], subspace_std)
    return makeup, bareface, Simg


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, barefaces, Simgs, makeups, vggloss, id_loss, device):
    loader = sample_data(loader)
    vgg_weights = [0.0, 0.5, 1.0, 0.0, 0.0]
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, smoothing=0.01, ncols=180, dynamic_ncols=False)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_bareface = torch.randn(args.n_sample, args.latent, device=device)
    sample_makeup, _, real_imgs = get_paired_data(barefaces, Simgs, makeups, batch_size=args.n_sample, random_ind=8)
    sample_makeup = sample_makeup.to(device)

    real_imgs = F.interpolate(real_imgs, 256)
    utils.save_image(
                        real_imgs,
                        f"log/{args.style}/beautybank-real.jpg",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

        
    for idx in pbar:
        i = idx + args.start_iter
        
        which = i % args.subspace_freq 

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        if which == 0:
            makeup, _, _ = get_paired_data(barefaces, Simgs, makeups, batch_size=args.batch, random_ind=8)
            makeup = makeup.to(device)
            bareface = mixing_noise(args.batch, args.latent, args.mixing, device)
            z_plus_latent = False
        else:
            makeup, bareface, real_img = get_paired_data(barefaces, Simgs, makeups, batch_size=args.batch, random_ind=8)
            makeup = makeup.to(device)
            bareface = [bareface.to(device)]
            real_img = real_img.to(device)
            z_plus_latent = True
            
        fake_img, _ = generator(bareface, makeup, use_res=True, z_plus_latent=z_plus_latent)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss 
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)
            
            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        if which == 0:
            makeup, _, real_img = get_paired_data(barefaces, Simgs, makeups, batch_size=args.batch, random_ind=8)
            real_img = real_img.to(device)
            makeup = makeup.to(device)
            bareface = mixing_noise(args.batch, args.latent, args.mixing, device)
            z_plus_latent = False
        else:
            makeup, bareface, real_img = get_paired_data(barefaces, Simgs, makeups, batch_size=args.batch, random_ind=8)
            makeup = makeup.to(device)
            bareface = [bareface.to(device)]
            real_img = real_img.to(device)
            z_plus_latent = True
            
        fake_img, _ = generator(bareface, makeup, use_res=True, z_plus_latent=z_plus_latent)
        
        with torch.no_grad():  
            real_img_256 = F.adaptive_avg_pool2d(real_img, 256).detach()
            real_feats = vggloss(real_img_256)
            real_styles = [F.adaptive_avg_pool2d(real_feat, output_size=1).detach() for real_feat in real_feats]
            real_content, _ = generator(bareface, None, use_res=False, z_plus_latent=z_plus_latent)
            real_content_256 = F.adaptive_avg_pool2d(real_content, 256).detach() 
        
        fake_img_256 = F.adaptive_avg_pool2d(fake_img, 256)
        fake_feats = vggloss(fake_img_256)
        fake_styles = [F.adaptive_avg_pool2d(fake_feat, output_size=1) for fake_feat in fake_feats]
        sty_loss = (torch.tensor(0.0).to(device) if args.CX_loss == 0 else 
                    FCX.contextual_loss(fake_feats[2], real_feats[2].detach(), 
                                        band_width=0.2, loss_type='cosine') * args.CX_loss)
        if args.style_loss > 0:
            sty_loss += ((F.mse_loss(fake_styles[1], real_styles[1]) 
                    + F.mse_loss(fake_styles[2], real_styles[2])) * args.style_loss)
        
        ID_loss = (torch.tensor(0.0).to(device) if args.id_loss == 0 else
                    id_loss(fake_img_256, real_content_256) * args.id_loss)
        
        gr_loss = torch.tensor(0.0).to(device)
        if which > 0:
            for ii, weight in enumerate(vgg_weights):
                if weight * args.perc_loss > 0:
                    gr_loss += F.l1_loss(fake_feats[ii], real_feats[ii].detach()) * weight * args.perc_loss
        
        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)
        l2_reg_loss = sum(torch.norm(p) for p in g_module.res.parameters()) * args.L2_reg_loss
        
        loss_dict["g"] = g_loss # Ladv
        loss_dict["gr"] = gr_loss # Lperc
        loss_dict["l2"] = l2_reg_loss # Lreg in Lcon
        loss_dict["id"] = ID_loss # LID in Lcon
        loss_dict["sty"] = sty_loss # Lsty
        g_loss = g_loss + gr_loss + sty_loss + l2_reg_loss + ID_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            
            bareface = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            makeup, _, _ = get_paired_data(barefaces, Simgs, makeups, batch_size=path_batch_size, random_ind=8)
            makeup = makeup.to(device)
            
            fake_img, latents = generator(bareface, makeup, return_latents=True, use_res=True, z_plus_latent=False)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema.res, g_module.res, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        gr_loss_val = loss_reduced["gr"].mean().item()
        sty_loss_val = loss_reduced["sty"].mean().item()
        l2_loss_val = loss_reduced["l2"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        id_loss_val = loss_reduced["id"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"iter: {i:d}; d: {d_loss_val:.3f}; g: {g_loss_val:.3f}; gr: {gr_loss_val:.3f}; sty: {sty_loss_val:.3f}; l2: {l2_loss_val:.3f}; id: {id_loss_val:.3f}; "
                    f"r1: {r1_val:.3f}; path: {path_loss_val:.3f}; mean path: {mean_path_length_avg:.3f}; "
                    f"augment: {ada_aug_p:.4f};"
                )
            )

            if i % 100 == 0 or (i+1) == args.iter:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_bareface], sample_makeup, use_res=True)
                    sample = F.interpolate(sample,256)
                    utils.save_image(
                        sample,
                        f"log/%s/beautybank-%06d.jpg"%(args.style, i),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    

            if ((i+1) >= args.save_begin and (i+1) % args.save_every == 0) or (i+1) == args.iter:
                torch.save(
                    {
                        "g_ema": g_ema.state_dict(),
                    },
                    f"%s/%s/%s-%06d.pt"%(args.model_path, args.style, args.model_name, i+1),
                )

                


if __name__ == "__main__":
    device = "cuda"

    parser = TrainOptions()
    args = parser.parse()
    if args.local_rank == 0:
        print('*'*98)
        
    if not os.path.exists("log/%s/"%(args.style)):
        os.makedirs("log/%s/"%(args.style))
    if not os.path.exists("%s/%s/"%(args.model_path, args.style)):
        os.makedirs("%s/%s/"%(args.model_path, args.style))    
        
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    generator = BeautyBank(args.size, args.latent, args.n_mlp, 
                             channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = BeautyBank(args.size, args.latent, args.n_mlp, 
                         channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        list(generator.res.parameters()) + list(generator.style.parameters()),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        
        if "g_optim" in ckpt:
            g_optim.load_state_dict(ckpt["g_optim"])
        if "d_optim" in ckpt:
            d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )  

    transform2 = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])

    dataset = MultiResolutionDataset(args.lmdb_path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="BeautyBank")
    
    ckpt = torch.load(args.encoder_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = args.encoder_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = True
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)
    encoder = pSp(opts).to(device).eval()
    encoder.latent_avg = encoder.latent_avg.to(device)
    vggloss = VGG19().to(device).eval()
    id_loss = id_loss.IDLoss(args.identity_path).to(device).eval()

    print('Encoder model successfully loaded!')
    
    bareface_dict = np.load(args.bareface_path, allow_pickle='TRUE').item()
    makeup_dict = np.load(args.makeup_path, allow_pickle='TRUE').item()
    path = args.image_path
    barefaces = []
    makeups = []
    Simgs = []
    for filename in bareface_dict.keys():   
        bareface = bareface_dict[filename]
        makeup = makeup_dict[filename]
        with torch.no_grad(): 
            makeup = torch.tensor(makeup).to(device)
            makeup = g_ema.generator.style(makeup[0]).reshape(makeup.shape).detach().cpu()
        Simg = Image.open(os.path.join(path, filename))
        Simg = transform2(Simg).unsqueeze(dim=0)
        barefaces += [bareface]
        makeups += [makeup]
        Simgs += [Simg]

    barefaces = torch.tensor(np.concatenate(barefaces, axis=0)) 
    Simgs = torch.cat(Simgs, dim=0)
    makeups = torch.tensor(np.concatenate(makeups, axis=0)) 
    print('Data successfully loaded!')
    
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, barefaces, Simgs, makeups, vggloss, id_loss, device)
    
    
    