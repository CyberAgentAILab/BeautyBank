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
        
        self.parser = argparse.ArgumentParser(description="Interpolation")
        self.parser.add_argument("--content", type=str, default='./data/test/003767.png', help="path of the bare-face image 1")
        self.parser.add_argument("--content2", type=str, default='./data/test/083311.png', help="path of the bare-face image 2")
        self.parser.add_argument("--style", type=str, default='makeup', help="target makeup style type")  
        self.parser.add_argument("--style_id", type=int, default=0, help="the id of the makeup style image") 
        self.parser.add_argument("--style_id2", type=int, default=1, help="the id of the makeup style image")
        self.parser.add_argument("--truncation", type=float, default=0.75, help="truncation for bare-face code (content)")
        self.parser.add_argument("--weight", type=float, nargs=18, default=[0.75]*7+[1]*11, help="weight of the makeup style")
        self.parser.add_argument("--name", type=str, default='makeup', help="filename to save the generated images")
        self.parser.add_argument("--preserve_color", action="store_true", help="preserve the color of the content image")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='generator.pt', help="name of the saved BeautyBank")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--align_face", action="store_true", help="apply face alignment to the content image")
        self.parser.add_argument("--makeup_name_1", type=str, default=None, help="name of the makeup codes")
        self.parser.add_argument("--makeup_name_2", type=str, default=None, help="name of the makeup codes")
        self.parser.add_argument("--wplus", action="store_true", help="use original pSp encoder to extract the bare-face code")

    def parse(self):
        self.opt = self.parser.parse_args()    
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

    #content 2
    aligned_image2 = align_face(filepath=args.content2, predictor=predictor)

    return aligned_image, aligned_image2


if __name__ == "__main__":
    device = "cuda"

    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
    
    transform = transforms.Compose([

    transforms.Resize(1024),
    transforms.CenterCrop(1024),

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

    makeups_1 = np.load(os.path.join(args.model_path, args.style, args.makeup_name_1), allow_pickle='TRUE').item()
    makeups_2 = np.load(os.path.join(args.model_path, args.style, args.makeup_name_2), allow_pickle='TRUE').item()

    z_plus_latent=not args.wplus
    return_z_plus_latent=not args.wplus
    input_is_latent=args.wplus    
    
    print('Load models successfully!')
    
    with torch.no_grad():
        viz = []

        # load content image 2
        if args.align_face:
            I1,I2 = run_alignment(args)
            I1 = transform(I1).unsqueeze(dim=0).to(device)
            I2 = transform(I2).unsqueeze(dim=0).to(device)
            I1 = F.adaptive_avg_pool2d(I1, 1024)
            I2 = F.adaptive_avg_pool2d(I2, 1024)
        else:
            I1 = load_image(args.content).to(device)
            I2 = load_image(args.content2).to(device)

        viz += [I1]
        viz += [I2]

        # reconstructed content image and its bare-face style code
        img_rec1, instyle1 = encoder(F.adaptive_avg_pool2d(I1, 256), randomize_noise=False, return_latents=True, 
                                   z_plus_latent=z_plus_latent, return_z_plus_latent=return_z_plus_latent, resize=False)  
        img_rec1 = torch.clamp(img_rec1.detach(), -1, 1)
        viz += [img_rec1]

        img_rec2, instyle2 = encoder(F.adaptive_avg_pool2d(I2, 256), randomize_noise=False, return_latents=True, 
                                   z_plus_latent=z_plus_latent, return_z_plus_latent=return_z_plus_latent, resize=False)  
        img_rec2 = torch.clamp(img_rec2.detach(), -1, 1)
        viz += [img_rec2]

        print("number of style 1 you have is : }" + str(len(list(makeups_1.keys()))))   
        print("number of style 2 you have is : }" + str(len(list(makeups_2.keys()))))   
        stylename1 = list(makeups_1.keys())[args.style_id]
        stylename2 = list(makeups_2.keys())[args.style_id2]

        latent1 = torch.tensor(makeups_1[stylename1]).to(device)
        latent2 = torch.tensor(makeups_2[stylename2]).to(device)

        if args.preserve_color and not args.wplus:
            latent1[:,7:18] = instyle1[:,7:18]
            latent2[:,7:18] = instyle2[:,7:18]

        # makeup style code
        makeup1 = generator.generator.style(latent1.reshape(latent1.shape[0]*latent1.shape[1], latent1.shape[2])).reshape(latent1.shape)
        makeup2 = generator.generator.style(latent2.reshape(latent2.shape[0]*latent2.shape[1], latent2.shape[2])).reshape(latent2.shape)
        if args.preserve_color and args.wplus:
            makeup1[:,7:18] = instyle1[:,7:18]
            makeup2[:,7:18] = instyle2[:,7:18]
            
        # load makeup style image if it exists
        S1 = None
        S2 = None
        if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename1)):
            S1 = load_image(os.path.join(args.data_path, args.style, 'images/train', stylename1)).to(device)
            S1 = F.interpolate(S1, size=(1024, 1024), mode='bilinear', align_corners=False) 
            viz += [S1]
            S2 = load_image(os.path.join(args.data_path, args.style, 'images/train', stylename2)).to(device)
            S2 = F.interpolate(S2, size=(1024, 1024), mode='bilinear', align_corners=False) 
            viz += [S2]

        # interpolation
        for inter in range(0, 5):

            instyle = instyle1 + (instyle2 - instyle1) * inter / 4
            makeup = makeup1 + (makeup2 - makeup1) * inter / 4

            img_gen, _ = generator([instyle], makeup, input_is_latent=input_is_latent, z_plus_latent=z_plus_latent,
                              truncation=args.truncation, truncation_latent=0, use_res=True, interp_weights=args.weight)
            img_gen = torch.clamp(img_gen.detach(), -1, 1)
            viz += [img_gen]
            print(inter)

    print('Generate images successfully!')
    for tensor in viz:
        print(tensor.shape)
    
    save_name = args.name+'_%d_%s'%(args.style_id, os.path.basename(args.content).split('.')[0])
    os.makedirs(os.path.join(args.output_path, args.style, 'interpolation'), exist_ok = True)
    print(os.path.join(args.output_path, args.style, 'interpolation'))
    save_image(torchvision.utils.make_grid(F.adaptive_avg_pool2d(torch.cat(viz, dim=0), 256), 5, 2).cpu(), 
               os.path.join(args.output_path, args.style, 'interpolation', save_name+'_overview.jpg'))
    save_image(img_gen[0].cpu(), os.path.join(args.output_path, args.style, 'interpolation', save_name+'.jpg'))

    print('Save images successfully!')
