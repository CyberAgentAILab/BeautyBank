import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from util import save_image, load_image
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
from model.BeautyBank import BeautyBank
from model.encoder.psp import pSp

class TestOptions():
    def __init__(self):
        
        self.parser = argparse.ArgumentParser(description="Face Generation")
        self.parser.add_argument("--content", type=str, default='./data/test/003767.png', help="path of the bare-face image")
        self.parser.add_argument("--style", type=str, default='makeup', help="target makeup style type")  
        self.parser.add_argument("--style_id", type=int, default=1, help="the id of the makeup style image") 
        self.parser.add_argument("--truncation", type=float, default=0.75, help="truncation for bare-face code (content)")
        self.parser.add_argument("--weight", type=float, nargs=18, default=[0.75]*7+[1]*11, help="weight of the makeup style")
        self.parser.add_argument("--name", type=str, default='generated_face', help="filename to save the generated images")
        self.parser.add_argument("--preserve_color", action="store_true", help="preserve the color of the content image")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='generator.pt', help="name of the saved BeautyBank")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--align_face", action="store_true", help="apply face alignment to the content image")
        self.parser.add_argument("--makeup_name", type=str, default=None, help="name of the makeup style codes")
        self.parser.add_argument("--wplus", action="store_true", help="use original pSp encoder to extract the bare-face code")

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.makeup_name is None:
            if os.path.exists(os.path.join(self.opt.model_path, self.opt.style, 'refined_makeup_code.npy')):
                self.opt.makeup_name = 'refined_makeup_code.npy'
            else:
                self.opt.makeup_name = 'makeup_code.npy'        
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
def run_alignment(args):
    import dlib
    from model.encoder.align_all_parallel import align_face
    modelname = os.path.join(args.model_path, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 
    predictor = dlib.shape_predictor(modelname)
    aligned_image = align_face(filepath=args.content, predictor=predictor)
    return aligned_image


if __name__ == "__main__":
    device = "cuda"

    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    generator = BeautyBank(1024, 512, 8, 2, res_index=6)
    generator.eval()

    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name), map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"])
    generator = generator.to(device)
    
    if args.wplus:
        model_path = os.path.join(args.model_path, 'encoder_wplus.pt')
    else:
        model_path = os.path.join(args.model_path, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    if 'output_size' not in opts:
        opts['output_size'] = 1024    
    opts = Namespace(**opts)
    opts.device = device
    encoder = pSp(opts)
    encoder.eval()
    encoder.to(device)

    makeups = np.load(os.path.join(args.model_path, args.style, args.makeup_name), allow_pickle='TRUE').item()

    z_plus_latent=not args.wplus
    return_z_plus_latent=not args.wplus
    input_is_latent=args.wplus    
    
    print('Load models successfully!')
    
    num_for_face = 10
    for num in range(20):
        with torch.no_grad():
            viz = []
            # load content image
            if args.align_face:
                I = transform(run_alignment(args)).unsqueeze(dim=0).to(device)
                I = F.adaptive_avg_pool2d(I, 1024)
            else:
                I = load_image(args.content).to(device)
            viz += [I]

            # reconstructed content image and its bare-face code
            img_rec, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True, 
                                    z_plus_latent=z_plus_latent, return_z_plus_latent=return_z_plus_latent, resize=False)  
            img_rec = torch.clamp(img_rec.detach(), -1, 1)
            viz += [img_rec]

            # Generate a random noise tensor with the same shape as instyle
            noise_instyle_origin = torch.randn_like(instyle).to(device)
            noise_instyle = noise_instyle_origin *0.3 + instyle  #0.1

            stylename = list(makeups.keys())[args.style_id]
            latent = torch.tensor(makeups[stylename]).to(device)
            if args.preserve_color and not args.wplus:
                latent[:,7:18] = noise_instyle[:,7:18]  
            # makeup code
            makeup = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)
            if args.preserve_color and args.wplus:
                makeup[:,7:18] = noise_instyle[:,7:18] 
                
            # load makeup style image if it exists
            S = None
            if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename)):
                S = load_image(os.path.join(args.data_path, args.style, 'images/train', stylename)).to(device)
                S = F.interpolate(S, size=(1024, 1024), mode='bilinear', align_corners=False) 
                viz += [S]

            # face generation
            img_gen, _ = generator([noise_instyle], makeup, input_is_latent=input_is_latent, z_plus_latent=z_plus_latent,
                                truncation=args.truncation, truncation_latent=0, use_res=True, interp_weights=args.weight)
            img_gen = torch.clamp(img_gen.detach(), -1, 1)
            viz += [img_gen]

        print('Generate images successfully!')

        print(args.output_path)
        
        save_name = args.name+'_%d_%s_%d'%(args.style_id, os.path.basename(args.content).split('.')[0], num)
        print("save name: " + save_name)
        os.makedirs(args.output_path, exist_ok = True)
        save_image(torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz, dim=0), 256), 4, 2).cpu(), 
                os.path.join(args.output_path, save_name+'_overview.jpg'))
        save_image(img_gen[0].cpu(), os.path.join(args.output_path, save_name+'.jpg'))

        print('Save images successfully!')
