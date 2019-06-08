import numpy as np
import subprocess
import os
from IPython.html.widgets import interact
from PIL import Image
import random
from PIL import ImageOps

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))

def align_eye_pad_ailab(im, anno, lambda1, lambda2, lambda3):
    p1 = np.array((anno['fm1x'], anno['fm1y'])).astype('f')
    p2 = np.array((anno['fm0x'], anno['fm0y'])).astype('f')
    face_width = anno['y2'] - anno['y1']
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    dis_width = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2.0
    pad_type = 'edge'
    pad_size = max(im.size[0], im.size[1]) / 2
    np_im = np.array(im)
    tmp_im = Image.fromarray(np.rot90(np.array([np.pad(np_im[:,:,0], pad_size, pad_type), \
                                      np.pad(np_im[:,:,1], pad_size, pad_type), \
                                      np.pad(np_im[:,:,2], pad_size, pad_type)]).T, 3))
    tmp_im = ImageOps.mirror(tmp_im)
    xc = xc + pad_size
    yc = yc + pad_size
    tmp_im = tmp_im.rotate(angle, center=(xc, yc), resample=Image.BICUBIC)
    w = face_width
    h = w / lambda1 * lambda2
    x1 = anno['y1'] - w / 2 + pad_size
    y1 = yc - w / lambda1 * lambda3
    x2 = x1 + 2*w
    y2 = y1 + h
    return tmp_im.crop((x1,y1,x2,y2)).resize((178,218), resample=Image.BICUBIC)

def face_detect(img_path):
    old_ld_library = os.environ['LD_LIBRARY_PATH']
    os.environ['LD_LIBRARY_PATH'] = './util/face/lib:./util/face/public_libs:/usr/local/caffe/lib:'+os.environ['LD_LIBRARY_PATH']
    tmp = subprocess.check_output(['./util/face/AILab_FaceDemo', img_path, './util/face/models/', '0'])
    os.environ['LD_LIBRARY_PATH'] = old_ld_library
    return eval('{'+''.join(tmp.split('\n')[1:16])+'}')

def align(img_path, attr, lambda1 = 77.0, lambda2 = 228.0, lambda3 = 111.0):
    im = Image.open(img_path).convert('RGB')
    im_aligned =align_eye_pad_ailab(im,attr, lambda1, lambda2, lambda3)
    return im_aligned

import glob
import random
import tqdm

img_paths = glob.glob('/data2/minjunli/prj/anime/fake_detect/haozhi_val/*')
img_paths += glob.glob('/data2/minjunli/prj/anime/fake_detect/filtered/*')
img_paths += glob.glob('/data2/minjunli/prj/anime/fake_detect/random/*')
random.shuffle(img_paths)
aligned_dir = '/data2/minjunli/prj/anime/fake_detect/aligned'
if not os.path.exists(aligned_dir):
    os.mkdir(aligned_dir)


import json

attr = {}
for img_path in tqdm.tqdm(img_paths):
    attr[img_path] = face_detect(img_path)
    # if len(attr) == 15:
        # img_name = img_path.split('/')[-1]
        # im_aligned = align(img_path, attr)
        # im_aligned.save(os.path.join(aligned_dir,img_name))

with open('data.json', 'w') as f:
            json.dump(attr, f)
        
