import argparse
import base64
import requests
import cStringIO
from io import BytesIO
import json
from PIL import Image
from PIL import ImageFilter

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

def predict_mask(img):
    image_width =  None
    image_height = None

    image_base64 = _to_img(img, image_width, image_height)

    r = requests.post(URL, json={"session_id": "xiaolongzhu", "img_data": image_base64})
    js = r.json()
    mask = _get_img(js['prob'], image_width, image_height)

    return mask
    

