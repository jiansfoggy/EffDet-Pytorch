import math
from random import shuffle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

def compute(img):
    R_mean = np.mean(img[:,:,0])   
    G_mean = np.mean(img[:,:,1])
    B_mean = np.mean(img[:,:,2])
    R_std = np.std(img[:,:,0])
    G_std = np.std(img[:,:,1])
    B_std = np.std(img[:,:,2])
    mean = (R_mean, G_mean, B_mean)
    std = (R_std, G_std, B_std)

    return mean, std

def preprocess_input(image):
    image /= 255
    mean, std = compute(image)
    #print(mean,std)
    #mean=(0.406, 0.456, 0.485)
    #std=(0.225, 0.224, 0.229)
    image -= mean
    image /= std
    return image

class EfficientdetDataset(Dataset):
    def __init__(self, train_lines, image_size, is_train):
        super(EfficientdetDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        #self.label_lines = label_lines
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, img_lst_path, lbl_lst_path, input_shape, jitter=.3, hue=1, sat=1.5, val=1.5, random=True):
        '''r实时数据增强的随机预处理'''
        #line = annotation_line.split()
        image = Image.open(img_lst_path)
        # what we get here is path for each image
        #image = cv2.imread(img_lst_path)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        iw, ih = image.size
        #iw, ih, channel = np.shape(image)
        h, w = input_shape
        
        #box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        with open(lbl_lst_path,'r') as f:
            Bboxes = f.read().strip().splitlines()
            Bbox = np.array([list(map(float,Bboxes[i].split(" "))) for i in range(len(Bboxes))])
            box = np.zeros((len(Bboxes), 5))
            box[:,:4] = Bbox[:,1:5]
            box[:, 4] = Bbox[:,0]
            box[:, 2] = box[:, 0] + box[:, 2]
            box[:, 3] = box[:, 1] + box[:, 3]

        if not random:
            # resize image
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
          
            #image = cv2.resize(image,(nw,nh), Image.BICUBIC)
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            #new_image.paste(image, (dx, dy, dx+nw, dy+nh))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box),5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box

            return image_data, box_data

        # resize image
        new_ar = w/h * self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        #image = cv2.resize(image,(nw,nh), Image.BICUBIC)

        # place image
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        #print(np.shape(new_image),np.shape(image))
        #print(dx,dy)
        # print(type(new_image),type(image))
        #new_image.paste(image, (0,0))
        image = new_image

        # flip image or not
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)

        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255 # numpy array, 0 to 1

        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box

        return image_data, box_data

    def __getitem__(self, index):
        #index = index % self.train_batches
        img_lst_path = self.train_lines[index]
        lbl_lst_path = img_lst_path.replace('images', 'labels').replace('.' + img_lst_path.split('.')[-1], '.txt')
        img, y = self.get_random_data(img_lst_path, lbl_lst_path, self.image_size[0:2], random=self.is_train)
        if len(y)!=0:
            boxes = np.array(y[:,:4],dtype=np.float32)
            y = np.concatenate([boxes,y[:,-1:]],axis=-1)

        img = np.array(img, dtype=np.float32)
        tmp_inp = np.transpose(preprocess_input(img),(2,0,1))
        tmp_targets = np.array(y, dtype=np.float32)
        return tmp_inp, tmp_targets

# DataLoader中collate_fn使用
def efficientdet_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    bboxes = np.array(bboxes)
    return images, bboxes

