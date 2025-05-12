import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch 
import argparse
from model.sampler.icp import ICPTrainer

class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Train Sampler")
        self.parser.add_argument("--style", type=str, default='makeup', help="target style type")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--makeup_path", type=str, default=None, help="path to the makeup codes")
        self.parser.add_argument("--model_name", type=str, default='sampler.pt', help="name of the saved model")

    def parse(self):
        self.opt = self.parser.parse_args()     
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

if __name__ == "__main__":
    device = "cuda"

    parser = TrainOptions()
    args = parser.parse()
    print('*'*50)
    
    if args.makeup_path is None:
        if os.path.exists(os.path.join(args.model_path, args.style, 'refined_makeup_code.npy')):
            makeups_dict = np.load(os.path.join(args.model_path, args.style, 'refined_makeup_code.npy'),allow_pickle='TRUE').item()
        else:
            makeups_dict = np.load(os.path.join(args.model_path, args.style, 'makeup_code.npy'),allow_pickle='TRUE').item()
    else:
        makeups_dict = np.load(args.makeup_path,allow_pickle='TRUE').item()
        
    makeups = []
    for k in makeups_dict.keys():
        makeups += [torch.tensor(makeups_dict[k])]
    makeups = torch.cat(makeups, dim=0).reshape(-1,18*512)
    
    # augment extrinsic style codes to about 1000 by duplicate and small jittering
    W = torch.normal(makeups.repeat(1000//makeups.shape[0], 1), 0.05)
    # color code
    WC = W[:,512*7:].detach().cpu().numpy()
    # style code
    WS = W[:,0:512*7].detach().cpu().numpy()
    
    print('Load makeup codes successfully!')
    # train color code sampler 
    icptc = ICPTrainer(WC, 128)
    icptc.icp.netT = icptc.icp.netT.to(device)
    icptc.train_icp(int(500000/WC.shape[0]))
    # train structure code sampler 
    icpts = ICPTrainer(WS, 128)
    icpts.icp.netT = icpts.icp.netT.to(device)
    icpts.train_icp(int(500000/WS.shape[0]))
    
    torch.save(
        {
            "color": icptc.icp.netT.state_dict(),
            "structure": icpts.icp.netT.state_dict(),
        },
        f"%s/%s/%s"%(args.model_path, args.style, args.model_name),
    )
    
    print('Training done!')
    
