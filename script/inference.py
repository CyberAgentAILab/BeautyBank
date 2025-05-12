import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'face-parsing.PyTorch'))
from model import BiSeNet

import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--output_path', required=True)
args = parser.parse_args()

def process_eyes(eyes_uint8, face):
    """Process eye regions by extending bounding box."""
    contours, _ = cv2.findContours(eyes_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure there are valid contours before proceeding
    if not contours:
        return eyes_uint8

    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    extended_h = int(h * 1.3)
    extended_y_bottom = min(y + extended_h, face.shape[0])

    # Iterate safely within array bounds
    for i in range(max(0, x), min(x + w, face.shape[1])):  
        for j in range(max(0, y), extended_y_bottom):
            if face[j, i]: 
                eyes_uint8[j, i] = 255

    return eyes_uint8



def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    # atts = [0 'background', 1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
    # 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_im = cv2.resize(vis_im, (256, 256), fx=stride, fy=stride, interpolation=cv2.INTER_LINEAR)
    im = cv2.resize(im, (256, 256), fx=stride, fy=stride, interpolation=cv2.INTER_LINEAR)

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, (256, 256), fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255



    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        im_name = save_path[:-4].split('/')[-1]
        vis_path = f'{args.output_path}/vis/'
        os.makedirs(vis_path, exist_ok=True)

        skin = (vis_parsing_anno == 1)[..., None]
        l_brow = (vis_parsing_anno == 2)[..., None]
        r_brow = (vis_parsing_anno == 3)[..., None]
        l_eye = (vis_parsing_anno == 4)[..., None]
        r_eye = (vis_parsing_anno == 5)[..., None]
        eye_g = (vis_parsing_anno == 6)[..., None]
        l_ear = (vis_parsing_anno == 7)[..., None]
        r_ear = (vis_parsing_anno == 8)[..., None]
        ear_r = (vis_parsing_anno == 9)[..., None]
        nose = (vis_parsing_anno == 10)[..., None]
        mouth = (vis_parsing_anno == 11)[..., None]
        u_lip = (vis_parsing_anno == 12)[..., None]
        l_lip = (vis_parsing_anno == 13)[..., None]
        neck = (vis_parsing_anno == 14)[..., None]
        neck_l = (vis_parsing_anno == 15)[..., None]

        face = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, r_eye, eye_g, nose, mouth, u_lip, l_lip))
        mouth = np.logical_or.reduce(( u_lip, l_lip ))
        r_eyes = np.logical_or.reduce((r_brow,r_eye ))
        l_eyes = np.logical_or.reduce((l_brow,l_eye ))
        no_eyes = np.logical_or.reduce((l_eye, r_eye ))

        #face
        face_save_path  = vis_path + '/face/'
        os.makedirs(face_save_path, exist_ok=True)
        face_uint8 = (face.astype(np.uint8) * 255)
        cv2.imwrite(face_save_path + f'{im_name}.png', face_uint8)

        #mouth
        mouth_save_path  = vis_path + '/mouth/'
        os.makedirs(mouth_save_path, exist_ok=True)
        mouth_uint8 = (mouth.astype(np.uint8) * 255)
        cv2.imwrite(mouth_save_path + f'{im_name}.png', mouth_uint8)

        #eyes
        eyes_save_path = vis_path + '/eyes/'
        os.makedirs(eyes_save_path, exist_ok=True)
        r_eyes_uint8 = (r_eyes.astype(np.uint8)  * 255)
        l_eyes_uint8 = (l_eyes.astype(np.uint8)  * 255)

        face_mask = face.astype(np.uint8) * 255
        r_eyes_uint8 = process_eyes(r_eyes_uint8, face_mask)
        l_eyes_uint8 = process_eyes(l_eyes_uint8, face_mask)

        no_eyes = (no_eyes.astype(np.uint8) * 255)
        no_eyes = np.squeeze(no_eyes)

        #process eye area
        combined_eyes = cv2.bitwise_or(r_eyes_uint8, l_eyes_uint8)
        combined_eyes = combined_eyes - no_eyes

        combined_save_path = vis_path + '/combined_eyes/'
        os.makedirs(combined_save_path, exist_ok=True)
        cv2.imwrite(combined_save_path + f'{im_name}.png', combined_eyes)
        
        face2 = np.logical_or.reduce((skin, l_brow, r_brow, l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, mouth, u_lip, l_lip, neck, neck_l))

        #foreground and background
        ##face2
        background_save_path  = vis_path + '/background/'
        os.makedirs(background_save_path, exist_ok=True)
        face_uint8 = (face2.astype(np.uint8) * 255)
        background = 255 - face_uint8

        ##gaussian
        kernel = 11
        blurred_background = cv2.GaussianBlur(background, (kernel, kernel), 0)

        foreground_save_path  = vis_path + '/foreground/'
        os.makedirs(foreground_save_path, exist_ok=True)
        blurred_foreground = 255 - blurred_background

        cv2.imwrite(background_save_path + f'{im_name}.png', blurred_background)
        cv2.imwrite(foreground_save_path + f'{im_name}.png', blurred_foreground)



def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    os.makedirs(respth, exist_ok=True)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        for image_path in tqdm.tqdm(os.listdir(dspth)):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)

            img = to_tensor(image).unsqueeze(0).cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))


if __name__ == "__main__":
    evaluate(dspth=args.data, cp=os.path.join(f'{os.path.abspath(os.path.dirname(__file__))}/res/cp', '79999_iter.pth'))


