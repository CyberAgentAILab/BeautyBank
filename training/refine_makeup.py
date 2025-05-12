import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch import optim
from util import save_image
import argparse
from torchvision import transforms
import torchvision
from PIL import Image
from tqdm import tqdm

from model.BeautyBank import BeautyBank
from model.stylegan import lpips
import model.contextual_loss.functional as FCX
from model.vgg import VGG19

class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Refine Makeup Codes")
        self.parser.add_argument("--style", type=str, default='makeup', help="target makeup style type")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--ckpt", type=str, default=None, help="path to the saved beautybank model")
        self.parser.add_argument("--makeup_path", type=str, default=None, help="path to the saved makeup codes")
        self.parser.add_argument("--bareface_path", type=str, default=None, help="path to the saved bareface codes")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--iter", type=int, default= 300, help="total training iterations")  
        self.parser.add_argument("--batch", type=int, default=1, help="batch size")
        self.parser.add_argument("--lr_color", type=float, default=0.01, help="learning rate for color parts")
        self.parser.add_argument("--lr_structure", type=float, default=0.005, help="learning rate for structure parts")
        self.parser.add_argument("--model_name", type=str, default='refined_makeup_code.npy', help="name to save the refined makeup codes")

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.ckpt is None:
            self.opt.ckpt = os.path.join(self.opt.model_path, self.opt.style, 'generator.pt') 
        if self.opt.makeup_path is None:
            self.opt.makeup_path = os.path.join(self.opt.model_path, self.opt.style, 'makeup_code.npy')    
        if self.opt.bareface_path is None:
            self.opt.bareface_path = os.path.join(self.opt.model_path, self.opt.style, 'bareface_code.npy')          
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

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

    parser = TrainOptions()
    args = parser.parse()
    print('*'*50)
    
    if not os.path.exists("log/%s/refine_makeup/"%(args.style)):
        os.makedirs("log/%s/refine_makeup/"%(args.style))
        
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
            # Optionally apply a threshold to ensure the mask is binary
            transforms.Lambda(lambda x: (x > 0).float())
            ])

    generator = BeautyBank(1024, 512, 8, 2, res_index=6).to(device)
    generator.eval()

    ckpt = torch.load(args.ckpt)
    generator.load_state_dict(ckpt["g_ema"])
    noises_single = generator.make_noise()

    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))
    vggloss = VGG19().to(device).eval()

    print('Load models successfully!')
    
    datapath = os.path.join(args.data_path, args.style, 'images/train')
    makeups_dict = np.load(args.makeup_path, allow_pickle='TRUE').item()
    barefaces_dict = np.load(args.bareface_path, allow_pickle='TRUE').item()
    files = list(makeups_dict.keys())

    print("The number of barefaces you have is : " + str(len(list(barefaces_dict.keys()))))
    print("The number of makeups you have is : " + str(len(list(makeups_dict.keys()))))

    dict = {}
    for ii in range(0,len(files),args.batch):
        batchfiles = files[ii:ii+args.batch]
        imgs = []
        makeups = []
        barefaces = []

        masks = [] 
        eye_masks = []
        mouth_masks = [] 

        for file in batchfiles:
            img = transform(Image.open(os.path.join(datapath, file)).convert("RGB"))
            imgs.append(img)
            makeups.append(torch.tensor(makeups_dict[file]))
            barefaces.append(torch.tensor(barefaces_dict[file]))

            #mask
            mask = mask_transform(Image.open(os.path.join('./data/makeup/masks/vis/face', file)).convert("RGB"))
            masks.append(mask)

            eye_mask = mask_transform(Image.open(os.path.join('./data/makeup/masks/vis/combined_eyes', file)).convert("RGB"))
            eye_masks.append(eye_mask)

            mouth_mask = mask_transform(Image.open(os.path.join('./data/makeup/masks/vis/mouth', file)).convert("RGB"))
            mouth_masks.append(mouth_mask)


        imgs = torch.stack(imgs, 0).to(device)
        masks = torch.stack(masks, 0).to(device) 
        eye_masks = torch.stack(eye_masks, 0).to(device) 
        mouth_masks = torch.stack(mouth_masks, 0).to(device) 

        makeups = torch.cat(makeups, dim=0).to(device)
        barefaces = torch.cat(barefaces, dim=0).to(device)
        with torch.no_grad():  
            real_feats = vggloss(imgs)

        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())
        for noise in noises:
            noise.requires_grad = True  
            
        # color code
        makeups_c = makeups[:,7:].detach().clone()
        makeups_c.requires_grad = True
        # structure code
        makeups_s = makeups[:,0:7].detach().clone()
        makeups_s.requires_grad = True

        optimizer = optim.Adam([{'params':makeups_c,'lr':args.lr_color}, 
                                {'params':makeups_s,'lr':args.lr_structure}, 
                                {'params':noises,'lr':0.1}])

        pbar = tqdm(range(args.iter), smoothing=0.01, dynamic_ncols=False, ncols=100)
        
        for i in pbar:       

            latent = torch.cat((makeups_s, makeups_c), dim=1)
            latent = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)

            img_gen, _ = generator([barefaces], latent, noise=noises, use_res=True, z_plus_latent=True)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            if i == 0:
                img_gen0 = img_gen.detach().clone()

            Lperc = percept(img_gen, imgs).sum()
            Lnoise = noise_regularize(noises)

            L1_mask = (torch.abs(img_gen - imgs) * masks).sum() 
            Lperc_unmasked = percept(img_gen * masks, imgs * masks) 
            Lperc_masked = Lperc_unmasked

            Lperc_eye_unmasked = percept(img_gen * eye_masks, imgs * eye_masks) 
            L_eye_masked = Lperc_eye_unmasked 
            Lperc_mouth_unmasked = percept(img_gen * mouth_masks, imgs * mouth_masks) 
            L_mouth_masked = Lperc_mouth_unmasked 


            fake_feats = vggloss(img_gen)
            LCX = FCX.contextual_loss(fake_feats[2], real_feats[2].detach(), band_width=0.2, loss_type='cosine')

            loss = Lperc + LCX + 1e5 * Lnoise + 1e-4 * L1_mask + 100 * Lperc_masked + 100 * L_eye_masked 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            noise_normalize_(noises)

            pbar.set_description(
                (
                    f"[{ii * args.batch:03d}/{len(files):03d}]"
                    f" Lp: {Lperc.item():.3f}; Lnoise: {Lnoise.item():.3f};"
                    f" LCX: {LCX.item():.3f};"
                    f" L1: {L1_mask.item():.3f};"         
                    f" Lpm: {Lperc_masked.item():.3f};"  
                    f" Le: {L_eye_masked.item():.3f};" 
                    f" Lm: {L_mouth_masked.item():.3f};"         
                )
            )


        with torch.no_grad():
            latent = torch.cat((makeups_s, makeups_c), dim=1)
            for j in range(imgs.shape[0]):
                vis = torchvision.utils.make_grid(torch.cat([imgs[j:j+1], masks[j:j+1], img_gen0[j:j+1], img_gen[j:j+1].detach()], dim=0), 4, 1)
                save_image(torch.clamp(vis.cpu(),-1,1), os.path.join("./log/%s/refine_makeup/"%(args.style), batchfiles[j]))
                dict[batchfiles[j]] = latent[j:j+1].cpu().numpy()

    np.save(os.path.join(args.model_path, args.style, args.model_name), dict) 
    
    print('Refinement done!')
    