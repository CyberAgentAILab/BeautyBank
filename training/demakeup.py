import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch import optim
from util import save_image
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
from PIL import Image
from tqdm import tqdm
import math

from model.stylegan.model import Generator
from model.stylegan import lpips
from model.encoder.psp import pSp
from model.encoder.criteria import id_loss

class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Facial Demakeup")
        self.parser.add_argument("--style", type=str, default="makeup", help="target style type")
        self.parser.add_argument("--truncation", type=float, default=0.5, help="truncation for bare-face (content)")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='finetune-000600.pt', help="name of the saved fine-tuned model")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--iter", type=int, default=300, help="total training iterations")
        self.parser.add_argument("--batch", type=int, default=1, help="batch size")


    def parse(self):
        self.opt = self.parser.parse_args()        
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss

def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)
        
if __name__ == "__main__":
    device = "cuda"

    parser = TestOptions()
    args = parser.parse()
    print('*'*50)
      
    os.makedirs(f"log/{args.style}/demakeup", exist_ok=True)

        
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    mask_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0).float())
            ])
            
    # fine-tuned StyleGAN g'
    generator_prime = Generator(1024, 512, 8, 2).to(device)
    generator_prime.eval()
    # orginal StyleGAN g
    generator = Generator(1024, 512, 8, 2).to(device)
    generator.eval()

    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name))
    generator_prime.load_state_dict(ckpt["g_ema"])
    ckpt = torch.load(os.path.join(args.model_path, 'stylegan2-ffhq-config-f.pt'))
    generator.load_state_dict(ckpt["g_ema"])
    noises_single = generator.make_noise()

    model_path = os.path.join(args.model_path, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    encoder = pSp(opts).to(device).eval()

    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))
    id_loss = id_loss.IDLoss(os.path.join(args.model_path, 'model_ir_se50.pth')).to(device).eval()

    print('Load models successfully!')
    
    datapath = os.path.join(args.data_path, args.style, 'images/train')
    files = os.listdir(datapath) 
    
    bareface_dict = {}
    makeup_dict = {}
    for ii in range(0,len(files),args.batch):

        batchfiles = files[ii:ii + args.batch]
        if not batchfiles:
            continue

        imgs = []
        masks = []
        for file in batchfiles:
            img = transform(Image.open(os.path.join(datapath, file)).convert("RGB"))
            imgs.append(img)

            #mask
            mask = mask_transform(Image.open(os.path.join('./data/makeup/masks/vis/face', file)).convert("RGB"))
            masks.append(mask)

        imgs = torch.stack(imgs, 0).to(device)
        masks = torch.stack(masks, 0).to(device)

        with torch.no_grad():  
            # reconstructed bare-face and makeup code
            img_rec, latent_e = encoder(imgs, randomize_noise=False, return_latents=True, z_plus_latent=True)
            
        for j in range(imgs.shape[0]):
            makeup_dict[batchfiles[j]] = latent_e[j:j+1].cpu().numpy()
        
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())
        for noise in noises:
            noise.requires_grad = True    

        latent = latent_e.detach().clone()
        latent.requires_grad = True

        optimizer = optim.Adam([latent] + noises, lr=0.1)

        pbar = tqdm(range(args.iter))

        for i in pbar:
            t = i / args.iter
            lr = get_lr(t, 0.1)
            optimizer.param_groups[0]["lr"] = lr
            latent_n = latent

            img_gen, _ = generator_prime([latent_n], input_is_latent=False, noise=noises, z_plus_latent=True)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            Lperc = percept(img_gen, imgs).sum()
            LID = id_loss(img_gen, imgs)
            Lreg = latent.std(dim=1).mean()
            Lnoise = noise_regularize(noises)
            L1_mask = (torch.abs(img_gen - imgs) * masks).sum() / masks.sum().item()

            loss = 1 * Lperc + 0.1 * LID + Lreg + 1e5 * Lnoise + 100 * L1_mask

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            pbar.set_description(
                (
                    f"[{ii:03d}/{len(files):03d}]"
                    f" Lperc: {Lperc.item():.3f}; Lnoise: {Lnoise.item():.3f};"
                    f" LID: {LID.item():.3f}; Lreg: {Lreg.item():.3f}; lr: {lr:.3f}; l1: {L1_mask.item():.3f}"
                )
            )

        with torch.no_grad():
            latent[:,8:18] = latent_e[:,8:18].detach()   
            img_dsty, _ = generator([latent.detach()], input_is_latent=False, truncation=args.truncation, 
                           truncation_latent=0, noise=noises, z_plus_latent=True)
            img_dsty = F.adaptive_avg_pool2d(img_dsty.detach(), 256)
            # (optinal) preserve color
            _, latent_i = encoder(img_dsty, randomize_noise=False, return_latents=True, z_plus_latent=True)

            latent_i[:,8:18] = latent_e[:,8:18].detach()   

            img_refine, _ = generator([latent_i.detach()], input_is_latent=False, truncation=args.truncation, 
                           truncation_latent=0, noise=noises, z_plus_latent=True)
            img_refine = F.adaptive_avg_pool2d(img_refine.detach(), (256,256))

            for j in range(imgs.shape[0]):
                vis = torchvision.utils.make_grid(torch.cat([imgs[j:j+1], img_rec[j:j+1].detach(), 
                                             img_dsty[j:j+1].detach()], dim=0), 3, 1)

                save_image(torch.clamp(vis.cpu(),-1,1), os.path.join("./log/%s/demakeup/"%(args.style), batchfiles[j]))
                bareface_dict[batchfiles[j]] = latent_i[j:j+1].cpu().numpy()
    
    np.save(os.path.join(args.model_path, args.style, 'bareface_code.npy'), bareface_dict)    
    np.save(os.path.join(args.model_path, args.style, 'makeup_code.npy'), makeup_dict) 
    print('Demakeup done!')
    
