import numpy as np
import subprocess
import os
from IPython.html.widgets import interact
from PIL import Image
import random
from PIL import ImageOps
import visualize_test

import argparse
import base64
import requests
import cStringIO
from io import BytesIO
import json
from PIL import Image

URL = 'http://100.102.36.11:30001'


def _to_img(img, image_width, image_height):
    i = img
    if image_width != None and image_height != None:
        i = i.resize((image_width, image_height))
    buf = cStringIO.StringIO()
    i.save(buf, format="JPEG")
    buf = base64.encodestring(buf.getvalue())
    buf += "=" * (-len(buf) % 4)
    q = buf
    # print(q)
    return q

def _get_img(img_base64, image_width, image_height):
    '''base64 to numpy'''
    #print(img_base64)
    i = Image.open(BytesIO(base64.decodestring(img_base64))).convert('RGB')
    #print(i.size)
    imw, imh = i.size
    if image_width != None and image_height != None:
        i = i.resize((image_width, image_height))
    return i

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
    x1 = anno['y1'] - w/2 + pad_size
    y1 = yc - w / lambda1 * lambda3
    x2 = x1 + 2*w
    y2 = y1 + h
    return tmp_im.crop((x1,y1,x2,y2)).resize((178,218), resample=Image.BICUBIC)

def align_eye_pad_ailab_show(im, anno, lambda1, lambda2, lambda3):
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
    x1 = anno['y1'] - w/2 + pad_size
    y1 = yc - w / lambda1 * lambda3
    xd = 2*w
    yd = h
    return tmp_im.crop((x1,y1+xd*0.2,x1+xd,y1+xd)).resize((128,128), resample=Image.BICUBIC)

def align_eye_pad_ailab_show2(im, anno, lambda1, lambda2, lambda3):
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
    x1 = anno['y1'] - w/2 + pad_size
    y1 = yc - w / lambda1 * lambda3
    xd = 2*w
    yd = h
    print xd, yd
    return tmp_im.crop((x1-(yd-xd)/2,y1,x1-(yd-xd)/2+yd,y1+yd)).resize((128,128), resample=Image.BICUBIC)

def face_detect(img_path, gpu_id=7):
    old_ld_library = os.environ['LD_LIBRARY_PATH']
    os.environ['LD_LIBRARY_PATH'] = './utils/face/lib:./utils/face/public_libs:/data2/minjunli/tol/cuda/lib64'#+os.environ['LD_LIBRARY_PATH']
    tmp = subprocess.check_output(['./utils/face/AILab_FaceDemo', img_path, './utils/face/models/', str(gpu_id)])
    os.environ['LD_LIBRARY_PATH'] = old_ld_library
    return eval('{'+''.join(tmp.split('\n')[1:16])+'}')

def predict_mask(img):
    image_width =  None
    image_height = None

    image_base64 = _to_img(img, image_width, image_height)

    r = requests.post(URL, json={"session_id": "xiaolongzhu", "img_data": image_base64})
    print r.status_code
    #print r.content
    js = r.json()
    #print js
    mask = _get_img(js['prob'], image_width, image_height)
    #image_matting = _get_img(js['img_data'], image_width, image_height)

    img_npy = np.asarray(img)

    mask = mask.point(lambda p: p > 50 and 255) 
    #mask = mask.point(lambda p: p < 128 or 0)

    mask_npy = np.asarray(mask) / 255

    mask_inv_npy = 1 - mask_npy
    white_img_npy = np.ones((img_npy.shape),dtype=np.uint8) * 255


    img_masked_npy = np.multiply(img_npy, mask_npy) + np.multiply(white_img_npy, mask_inv_npy)
    img_masked = Image.fromarray(img_masked_npy)
    return img_masked



from PIL import ImageFilter
def predict_mask_alt(img):
    image_width =  None
    image_height = None

    image_base64 = _to_img(img, image_width, image_height)

    r = requests.post(URL, json={"session_id": "xiaolongzhu", "img_data": image_base64})
    print r.status_code
    #print r.content
    js = r.json()
    #print js
    mask = _get_img(js['prob'], image_width, image_height)
    #image_matting = _get_img(js['img_data'], image_width, image_height)
    img_npy = np.asarray(img)
    white_img_npy = np.ones((img_npy.shape),dtype=np.uint8) * 255
    softmask = (1-np.array(mask.filter(ImageFilter.GaussianBlur))/255.0)
    masked_img = img * (1-softmask) + white_img_npy * (softmask)

    return Image.fromarray(masked_img.astype(np.uint8))

import math

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )

def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point
    radius = np.radians(angle)
    qx = ox + math.cos(radius) * (px - ox) - math.sin(radius) * (py - oy)
    qy = oy + math.sin(radius) * (px - ox) + math.cos(radius) * (py - oy)
    return qx, qy

def draw_point(draw, p, r=20, fill=(255,0,0)):
    if p is None:
        return
    x, y = p
    draw.ellipse(((x-r), (y-r), (x+r), (y+r)), fill=fill)

def angle_between_2_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    tan = (y2 - y1) / (x2 - x1)
    return np.degrees(np.arctan(tan))
    
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y    

def point_in_rect(p, p1, p2):
    return p[0] <= p2[0] and p[0] >= p1[0] and  p[1] <= p2[1] and p[1] >= p1[1]

def align_eye_pad_ailab_show_alt(im, anno, lambda1 = 77.0, lambda2 = 228.0, lambda3 = 111.0, zoom=0.1, show=True):
    if 'fm1x' not in anno.keys():
        im
    p1 = np.array((anno['fm1x'], anno['fm1y'])).astype('f')
    p2 = np.array((anno['fm0x'], anno['fm0y'])).astype('f')
    face_width = anno['y2'] - anno['y1']
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    dis_width = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2.0
    pad_type = 'constant'
    pad_size = max(im.size[0], im.size[1]) / 2
    np_im = np.array(im)
    tmp_im = Image.fromarray(np.rot90(np.array([np.pad(np_im[:,:,0], pad_size, pad_type, constant_values=255), \
                                      np.pad(np_im[:,:,1], pad_size, pad_type, constant_values=255), \
                                      np.pad(np_im[:,:,2], pad_size, pad_type, constant_values=255)]).T, 3))
    tmp_im = ImageOps.mirror(tmp_im)
    xc = xc + pad_size
    yc = yc + pad_size
    tmp_im = tmp_im.rotate(angle, center=(xc, yc), resample=Image.BICUBIC)
    
    w = face_width
    h = w / lambda1 * lambda2
    x1 = anno['y1'] - w/2 + pad_size
    y1 = yc - w / lambda1 * lambda3
    xd = 2*w 
    yd = h
    adj = yd * zoom

    if show:
        crop_x1 = (x1)-((yd-xd)/2)+adj
        crop_y1 = y1+adj
        crop_x2 = (x1)-((yd-xd)/2)+yd-adj
        crop_y2 = y1+yd-adj
    else:
        crop_x1 = x1
        crop_y1 = y1
        crop_x2 = x1+xd
        crop_y2 = y1+yd
    
    rotated_anno = {}
    for i in range(0,5):
        rotated_anno['fm%dx'%i], rotated_anno['fm%dy'%i] = rotate_point((xc, yc), (anno['fm%dx'%i] + pad_size, anno['fm%dy'%i] + pad_size), -angle)

    crop_1 = (crop_x1, crop_y1)
    crop_2 = (crop_x2, crop_y1)
    crop_3 = (crop_x1, crop_y2)
    crop_4 = (crop_x2, crop_y2)
    
    orig_1 = rotate_point((xc, yc), (pad_size, pad_size), -angle)
    orig_2 = rotate_point((xc, yc), (pad_size + im.size[0], pad_size), -angle)
    orig_3 = rotate_point((xc, yc), (pad_size, pad_size + im.size[1]), -angle)
    orig_4 = rotate_point((xc, yc), (pad_size + im.size[0], pad_size + im.size[1]), -angle)
    
    center = (xc, yc)

    intersect = []
    dist = 999999.0
    nearest_point = None
    for i,j in [(1,2), (2,4), (1,3), (3,4)]:
        for k in (i,j):
            intersect.append((k,eval('line_intersection((orig_%d, orig_%d), (crop_%d, center))'%(i,j,k))))
            d = distance(intersect[-1][-1], center)
            if d < dist:
                dist = d
                nearest_point = intersect[-1]
    
    small_crop = False
    for i in intersect:
        if point_in_rect(i[-1], (crop_x1, crop_y1), (crop_x2, crop_y2)):
            small_crop = True
            break

    if nearest_point[0] == 1:
        x_tmp = crop_2
        y_tmp = crop_3
    if nearest_point[0] == 2:
        x_tmp = crop_1
        y_tmp = crop_4
    if nearest_point[0] == 3:
        x_tmp = crop_4
        y_tmp = crop_1
    if nearest_point[0] == 4:
        x_tmp = crop_3
        y_tmp = crop_2
    
    x2, _ = line_intersection((nearest_point[1], (nearest_point[1][0]+1, nearest_point[1][1])), (x_tmp, center))
    _, y2 = line_intersection((nearest_point[1], (nearest_point[1][0], nearest_point[1][1]+1)), (y_tmp, center))

    final_crop_x1 = min(nearest_point[1][0], x2)
    final_crop_y1 = min(nearest_point[1][1], y2)
    final_crop_x2 = max(nearest_point[1][0], x2)
    final_crop_y2 = max(nearest_point[1][1], y2)
    
    for i in range(0,5):
        if not point_in_rect((rotated_anno['fm%dx'%i],rotated_anno['fm%dy'%i]), (final_crop_x1, final_crop_y1), (final_crop_x2, final_crop_y2)):
            print 'Too Close to Boundary'
            tmp_im = Image.fromarray(np.array(tmp_im) / 4)
            break
    
    crop_pos = []
    x_ratio = 128.0 / (crop_x2 - crop_x1)
    y_ratio = 128.0 / (crop_y2 - crop_y1)
    crop_pos.append([(final_crop_x1 - crop_x1) * x_ratio, (final_crop_y1 - crop_y1) * y_ratio])
    crop_pos.append([(final_crop_x2 - crop_x1) * x_ratio, (final_crop_y2 - crop_y1) * y_ratio])
    
    orig_pos = []
    orig_pos.append(((orig_1[0] - crop_x1) * x_ratio, (orig_1[1] - crop_y1) * y_ratio))
    orig_pos.append(((orig_2[0] - crop_x1) * x_ratio, (orig_2[1] - crop_y1) * y_ratio))
    orig_pos.append(((orig_4[0] - crop_x1) * x_ratio, (orig_4[1] - crop_y1) * y_ratio))
    orig_pos.append(((orig_3[0] - crop_x1) * x_ratio, (orig_3[1] - crop_y1) * y_ratio))
#     print crop_pos
#     print orig_pos

   # return crop_pos, orig_pos, tmp_im.crop((crop_x1,crop_y1,crop_x2,crop_y2)).resize((128,128), resample=Image.BICUBIC)
    
    if not small_crop:
        return crop_pos, tmp_im.crop((crop_x1,crop_y1,crop_x2,crop_y2)).resize((128,128), resample=Image.BICUBIC)
    else:
        return crop_pos, tmp_im.crop((final_crop_x1,final_crop_y1,final_crop_x2,final_crop_y2)).resize((128,128), resample=Image.BICUBIC)

def post_process(img, orig_pos=None, crop_pos=None, col_add=0, row_del=0):
    t = img
    to_show_pre = transforms.Compose([transforms.Scale((128,128), Image.BICUBIC)
                                      ,transforms.ToTensor()
                                      ,transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
    if orig_pos is not None:
        mask = Image.new('RGB', (128,128), (255,255,255))
        draw = ImageDraw.Draw(mask)
        draw.polygon(orig_pos, fill=(0,0,0))
        del draw
        t[img_pre_128(mask)==1.0]=1.0
    if crop_pos is not None:
        crop_pos[0][0] = max(crop_pos[0][0], 0)
        crop_pos[0][1] = max(crop_pos[0][1], 0)
        crop_pos[1][0] = min(crop_pos[1][0], 128)
        crop_pos[1][1] = min(crop_pos[1][1], 128)
        padding = int((crop_pos[1][0] - crop_pos[0][0]) * 0.04)
        x1 = max(int(crop_pos[0][0]) - padding, 0)
        x2 = min(int(crop_pos[1][0]) + padding, 128)
        y1 = max(int(crop_pos[0][1]) + padding, 0)
        y2 = min(int(crop_pos[1][1]) - padding, 128)
        t = t[::,y1:y2,x1:x2]
    else:
        print 'orig'
        crop_pos = [[0,0],[128,128]]
        padding = int((crop_pos[1][0] - crop_pos[0][0]) * 0.04)
        x1 = max(int(crop_pos[0][0]) - padding, 0)
        x2 = min(int(crop_pos[1][0]) + padding, 128)
        y1 = max(int(crop_pos[0][1]) + padding, 0)
        y2 = min(int(crop_pos[1][1]) - padding, 128)
        t = t[::,y1:y2,x1:x2]
#         padding = int(128 * 0.07)
#         x1 = padding
#         x2 = 128 - padding
#         y1 = 0
#         y2 = 128
#         print x1, x2, y1, y2
#         t = t[::,y1:y2,x1:x2]
    if col_add != 0:
        tmp = torch.ones(3, 128, col_add)
        t = torch.cat((tmp, t, tmp), 2)
    if row_del != 0:
        t = t[::,row_del:(128-row_del)]
    return Image.fromarray(util.tensor2im(t.unsqueeze(0))).resize((256,256),Image.LANCZOS).resize((128,128),Image.LANCZOS)


def align(img, attr, lambda1 = 77.0, lambda2 = 228.0, lambda3 = 111.0, show=False):
    if not show:
        img_aligned =align_eye_pad_ailab(img,attr, lambda1, lambda2, lambda3)
    else:
        img_aligned =align_eye_pad_ailab_show_alt(img,attr, lambda1, lambda2, lambda3)
    return img_aligned


def align_eye_pad_ailab_full(im, anno, lambda1 = 77.0, lambda2 = 228.0, lambda3 = 111.0, pad_type='edge'):
    p1 = np.array((anno['fm1x'], anno['fm1y'])).astype('f')
    p2 = np.array((anno['fm0x'], anno['fm0y'])).astype('f')
    face_width = anno['y2'] - anno['y1']
    angle = angle_between_2_points(p1, p2)
    x1, y1 = p1
    x2, y2 = p2
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    dis_width = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2.0
    pad_size = max(im.size[0], im.size[1]) / 2
    np_im = np.array(im)
    if pad_type=='constant':
        tmp_im = Image.fromarray(np.rot90(np.array([np.pad(np_im[:,:,0], pad_size, pad_type, constant_values=255), \
                                          np.pad(np_im[:,:,1], pad_size, pad_type, constant_values=255), \
                                          np.pad(np_im[:,:,2], pad_size, pad_type, constant_values=255)]).T, 3))
    else:
        tmp_im = Image.fromarray(np.rot90(np.array([np.pad(np_im[:,:,0], pad_size, pad_type ), \
                                  np.pad(np_im[:,:,1], pad_size, pad_type), \
                                  np.pad(np_im[:,:,2], pad_size, pad_type)]).T, 3))
    tmp_im = ImageOps.mirror(tmp_im)
    xc = xc + pad_size
    yc = yc + pad_size
    tmp_im = tmp_im.rotate(angle, center=(xc, yc), resample=Image.BICUBIC)
    w = face_width
    h = w / lambda1 * lambda2
    x1 = anno['y1'] - w/2 + pad_size
    y1 = yc - w / lambda1 * lambda3
    x2 = x1 + 2*w
    y2 = y1 + h
    
    extent = 1.0
    nx1 = x1 - (x2 - x1) * extent / 2
    ny1 = y1 - (y2 - y1) * extent / 2
    nx2 = x2 + (x2 - x1) * extent / 2
    ny2 = y2 + (y2 - y1) * extent / 2
    print nx2 - nx1, x2 - x1
    print ny2 - ny1, y2 - y1
    tmp = 2.0
    return tmp_im.crop((nx1,ny1,nx2,ny2)).resize((int(128*tmp),int(128*tmp)), resample=Image.BICUBIC)

