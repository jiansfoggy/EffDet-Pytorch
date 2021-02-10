#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import colorsys
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from tqdm import tqdm

from efficientdet import EfficientDet
from nets.efficientdet import EfficientDetBackbone
from utils.utils import (bbox_iou, decodebox, efficientdet_correct_boxes,
                         letterbox_image, non_max_suppression)

image_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

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
    
    #mean=(0.406, 0.456, 0.485)
    #std=(0.225, 0.224, 0.229)
    image -= mean
    image /= std
    return image
    
class mAP_EfficientDet(EfficientDet):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        self.confidence = 0.35
        self.iou = 0.35
        #f = open("./input/detection-results/"+image_id+".txt","a") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = np.array(letterbox_image(image, (image_sizes[self.phi], image_sizes[self.phi])))
        photo = np.array(crop_img,dtype = np.float32)
        photo = np.transpose(preprocess_input(photo), (2, 0, 1))

        with torch.no_grad():
            images = torch.from_numpy(np.asarray([photo]))
            if self.cuda:
                images = images.cuda()

            #---------------------------------------------------------#
            #   传入网络当中进行预测
            #---------------------------------------------------------#
            _, regression, classification, anchors = self.net(images)
            print(len(classification))
            #-----------------------------------------------------------#
            #   将预测结果进行解码
            #-----------------------------------------------------------#
            regression = decodebox(regression, anchors, images)
            detection = torch.cat([regression,classification],axis=-1)
            batch_detections = non_max_suppression(detection, len(self.class_names),
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)
            #--------------------------------------#
            #   如果没有检测到物体，则返回原图
            #--------------------------------------#
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return 
            
            print(len(batch_detections))    
            #-----------------------------------------------------------#
            #   筛选出其中得分高于confidence的框 
            #-----------------------------------------------------------#
            top_index = batch_detections[:,4] > self.confidence
            top_conf = batch_detections[top_index,4]
            top_label = np.array(batch_detections[top_index,-1], np.int32)
            top_bboxes = np.array(batch_detections[top_index,:4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

            #-----------------------------------------------------------#
            #   去掉灰条部分
            #-----------------------------------------------------------#
            boxes = efficientdet_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([image_sizes[self.phi],image_sizes[self.phi]]),image_shape)
        
        #print(image_id)
        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = str(top_conf[i])
            top, left, bottom, right = boxes[i]
            #print("6 para: ", c, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom)))
            if score[:6] and str(left) and str(top) and str(right) and str(bottom) and str(c):
                with open("./input/detection-results/"+image_id+".txt","a") as f:
                    f.write("%s %s %s %s %s %s\n" % (c, score[:6], str(left), str(top), str(right), str(bottom)))
                    #f.write("%s %s %s %s %s %s\n" % (c, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
                    
        return 

efficientdet = mAP_EfficientDet()
image_ids = open('/media/data3/jian/Text_Detection/YOLOV3/valid.txt').read().strip().split()
if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

for image_id in tqdm(image_ids):
    #image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image_path = image_id
    image_id = image_id.split("/")[-1].split(".")[0]
    image = Image.open(image_path)
    # 开启后在之后计算mAP可以可视化
    # image.save("./input/images-optional/"+image_id+".jpg")
    efficientdet.detect_image(image_id,image)
    
path = './input/detection-results'    # 输入文件夹地址
files = os.listdir(path)   # 读入文件夹
print("%d images are detected text." %len(files))

print("Conversion completed!")
