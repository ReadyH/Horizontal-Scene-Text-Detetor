import os
import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from model import Model
from datautils import DataEncoder
import config
from utils import recover3, recover5

transform = transforms.ToTensor()

def get_img(img_dir):
    img_files = list()
    for (path, dir, files) in os.walk(img_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]

            if ext == '.jpg':
                img_files.append(path+'/'+filename)
    return img_files

def predict(img_dir, result_img_dir, result_txt_dir):

    minimum_loss = float('inf')
    minimum_idx = 0
    
    if torch.cuda.is_available():
        load_pth = torch.load('./model/model.pth')
    else:
        load_pth = torch.load('./model/model.pth', map_location=lambda storage, loc: storage)

    train_loss = load_pth['loss']

    model = Model()
    model.load_state_dict(load_pth['model'])
    model.eval()

    if not os.path.isdir(result_img_dir):
        os.mkdir(result_img_dir)
    if not os.path.isdir(result_txt_dir):   
        os.mkdir(result_txt_dir)
    encoder = DataEncoder()
    img_files = get_img(img_dir)
    for img_file in img_files:
        img_ori = Image.open(img_file)
        img = Image.open(img_file)
        w = h = config.input_image_size
        img = img.resize((w,h))
        
        image = transform(img)
        image = image.unsqueeze(0)
        loc_preds, cls_preds, mask_pred = model(image)
        loc_preds, cls_preds, mask_pred = loc_preds.to(torch.device('cpu')), cls_preds.to(torch.device('cpu')), mask_pred.to(torch.device('cpu'))
        boxes = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))
        if boxes is not None:
            boxes = recover3(img_ori, boxes)
        draw = ImageDraw.Draw(img_ori)

        img_file_name = img_file.split("/")[-1]
        txt_file_name = img_file_name.replace(".jpg", ".txt")
        txt_file_name = 'res_' + txt_file_name

        result_txt = open(result_txt_dir+"/"+txt_file_name, 'w')
        if boxes is not None:
            for box in boxes:
                draw.rectangle(list(box), outline='red')
                result_txt.write('%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\r\n' % (box[0].item(), box[1].item(), box[2].item(), box[1].item(), box[2].item(), box[3].item(), box[0].item(),box[3].item()))
        result_txt.close()

        img_ori.save(result_img_dir+"/"+img_file_name)

    return 'Done'