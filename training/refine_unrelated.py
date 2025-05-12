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

from model.BeautyBank import BeautyBank
from model.stylegan import lpips
from model.encoder.psp import pSp
import model.contextual_loss.functional as FCX
from model.vgg import VGG19

class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Disentangle Non-Makeup Features")
        self.parser.add_argument("--style", type=str, default='makeup', help="target style type")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--ckpt", type=str, default=None, help="path to the saved beautybank model")
        self.parser.add_argument("--makeup_path", type=str, default=None, help="path to the saved makeup codes")
        self.parser.add_argument("--bareface_path", type=str, default=None, help="path to the bareface images")
        self.parser.add_argument("--bareface_mask_path", type=str, default=None, help="path to the masks of bareface images")
        self.parser.add_argument("--makeup_file", type=str, default=None, help="path to the makeup file")
        self.parser.add_argument("--bareface_file", type=str, default=None, help="path to the bareface file")
        self.parser.add_argument("--iter", type=int, default= 300, help="total training iterations")  
        self.parser.add_argument("--batch", type=int, default=1, help="batch size")
        self.parser.add_argument("--lr_color", type=float, default=0.01, help="learning rate for color parts")
        self.parser.add_argument("--lr_structure", type=float, default=0.005, help="learning rate for structure parts")
        self.parser.add_argument("--model_name", type=str, default='refined_makeup_code.npy', help="name to save the refined makeup codes")
        self.parser.add_argument("--wplus", action="store_true", help="use original pSp encoder to extract the bareface code")

        self.parser.add_argument("--output_file", type=str, default='refine_makeup', help="file to save images")

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.ckpt is None:
            self.opt.ckpt = os.path.join(self.opt.model_path, self.opt.style, 'generator.pt') 
        if self.opt.makeup_path is None:
            self.opt.makeup_path = os.path.join(self.opt.model_path, self.opt.style, 'refined_makeup_code.npy')    
        if self.opt.bareface_path is None:
            self.opt.bareface_path = './data/makeup/images/test/'       
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

    #load psp
    if args.wplus:
        model_path = os.path.join(args.model_path, 'encoder_wplus.pt')
    else:
        model_path = os.path.join(args.model_path, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts['output_size'] = 1024
    opts = Namespace(**opts)
    opts.device = device 
    encoder = pSp(opts)
    encoder.eval()
    encoder.to(device)

    print('Load models successfully!')
    
    makeups_dict = np.load(args.makeup_path, allow_pickle='TRUE').item()
    files_style  =  list(makeups_dict.keys())

    bareface_dir = args.bareface_path  
    files = sorted([f for f in os.listdir(bareface_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(files)

    dict = {}
    imgs = []
    barefaces = []

    bf_file = args.bareface_file

    for i_file in range(len(files)):  

        if files[i_file] !=  bf_file:
                continue
        else:
            print("found " + files[i_file])
        
        imgs = []
        barefaces = []
        
        background_masks = []
        foregrond_masks = []
        face_masks = []
        eye_masks = []
        mouth_masks = [] 
        
        # mask
        mask_base = os.path.join(args.bareface_mask_path, 'vis')

        background_mask = mask_transform(
            Image.open(os.path.join(mask_base, 'background', files[i_file])).convert("RGB")
        )
        background_masks.append(background_mask)

        foregrond_mask = mask_transform(
            Image.open(os.path.join(mask_base, 'foreground', files[i_file])).convert("RGB")
        )
        foregrond_masks.append(foregrond_mask)

        face_mask = mask_transform(
            Image.open(os.path.join(mask_base, 'face', files[i_file])).convert("RGB")
        )
        face_masks.append(face_mask)

        eye_mask = mask_transform(
            Image.open(os.path.join(mask_base, 'combined_eyes', files[i_file])).convert("RGB")
        )
        eye_masks.append(eye_mask)

        mouth_mask = mask_transform(
            Image.open(os.path.join(mask_base, 'mouth', files[i_file])).convert("RGB")
        )
        mouth_masks.append(mouth_mask)

        img = transform(Image.open(os.path.join(args.bareface_path,files[i_file])).convert("RGB"))
        imgs.append(img)
        imgs = torch.stack(imgs, 0).to(device)

        #generate bareface code
        z_plus_latent=not args.wplus
        return_z_plus_latent=not args.wplus
        input_is_latent=args.wplus 

        img = img.unsqueeze(dim=0) 
        img_rec, bareface = encoder(F.adaptive_avg_pool2d(imgs, 256), randomize_noise=False, return_latents=True, 
                                   z_plus_latent=z_plus_latent, return_z_plus_latent=return_z_plus_latent, resize=False)  

        barefaces.append(bareface)
        
        background_masks = torch.stack(background_masks, 0).to(device) 
        foregrond_masks = torch.stack(foregrond_masks, 0).to(device) 
        face_masks = torch.stack(face_masks, 0).to(device) 
        eye_masks = torch.stack(eye_masks, 0).to(device) 
        mouth_masks = torch.stack(mouth_masks, 0).to(device) 
        
        barefaces = torch.cat(barefaces, dim=0).to(device)

        m_file = args.makeup_file

        for e_file in range(len(files_style)):
            if files_style[e_file] != m_file:
                continue
            else:
                print("found " + files_style[e_file])

            imgs_style = []
            makeups = []

            img_style = transform(Image.open(os.path.join('./data', args.style,'images/train', files_style[e_file])).convert("RGB"))
            imgs_style.append(img_style)
            makeups.append(torch.tensor(makeups_dict[files_style[e_file]]))

            imgs_style = torch.stack(imgs_style, 0).to(device) 
            makeups = torch.cat(makeups, dim=0).to(device)
            
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

                Lnoise = noise_regularize(noises)
                Lperc_fore = percept(img_gen * foregrond_masks, img_gen0 * foregrond_masks).sum()
                Lperc_back = percept(img_gen * background_masks, imgs * background_masks).sum()

                L1_mask = (torch.abs(img_gen - img_gen0) * face_masks).sum() 
                Lperc_unmasked = percept(img_gen * face_masks, img_gen0 * face_masks) 
                Lperc_face_masked = Lperc_unmasked 

                Lperc_eye_unmasked = percept(img_gen * eye_masks, img_gen0 * eye_masks) 
                L_eye_masked = Lperc_eye_unmasked 
                Lperc_mouth_unmasked = percept(img_gen * mouth_masks, img_gen0 * mouth_masks) 
                L_mouth_masked = Lperc_mouth_unmasked  

                real_feats = vggloss(imgs)
                fake_feats = vggloss(img_gen)
                LCX = FCX.contextual_loss(fake_feats[2], real_feats[2].detach(), band_width=0.2, loss_type='cosine')

                loss =  Lperc_fore + 100 * Lperc_back + 1e-4 * L1_mask + 0.1 * LCX + 1e5 * Lnoise + 100 * Lperc_face_masked + 200 * L_eye_masked + 200 * L_mouth_masked

                optimizer.zero_grad()

                loss.backward(retain_graph=True)

                optimizer.step()
                noise_normalize_(noises)

                pbar.set_description(
                    (
                        f"[{i_file * args.batch:03d}/{len(files):03d}]"
                        f" LCX: {LCX.item():.3f};"
                        f" L1: {L1_mask.item():.3f};"         
                        f" Lpf: {Lperc_fore.item():.3f};"  
                        f" Lpb: {Lperc_back.item():.3f};"   
                        f" Lnoise: {Lnoise.item():.3f};"   
                    )
                    +  
                    (
                        f" Lpf: {Lperc_face_masked.item():.3f};"   
                        f" Le: {L_eye_masked.item():.3f};" 
                        f" Lm: {L_mouth_masked.item():.3f};"              
                    )
                )
                        


            with torch.no_grad():
                latent = torch.cat((makeups_s, makeups_c), dim=1)
                for j in range(imgs.shape[0]):
                    vis = torchvision.utils.make_grid(torch.cat([imgs[j:j+1], imgs_style[j:j+1], foregrond_masks[j:j+1], img_gen0[j:j+1], img_gen[j:j+1].detach()], dim=0), 5, 1)
                    os.makedirs(os.path.join("./log/%s/"%(args.style + '/'+ args.output_file)), exist_ok=True)
                    name =  files[i_file].split('.')[0] + '_' +  files_style[e_file].split('.')[0] + ".png"
                    os.makedirs(os.path.join("./log/%s/"%(args.style + '/'+ args.output_file)), exist_ok=True)
                    save_image(torch.clamp(vis.cpu(),-1,1), os.path.join("./log/%s/"%(args.style + '/'+ args.output_file), name))
                    dict[files_style[e_file]] = latent[j:j+1].cpu().numpy()
                    #save makeup code 
                    style_name = name.split('.')[0] + "_" + args.model_name
                    os.makedirs(os.path.join(args.model_path, args.style, 'refine_weight'), exist_ok=True)
                    np.save(os.path.join(args.model_path, args.style, 'refine_weight',style_name), dict) 
            
            print('Refinement done for file ' + files[i_file] + ' and style ' + files_style[e_file])
    
    print("Refinement done!")