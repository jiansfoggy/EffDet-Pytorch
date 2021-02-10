#!/bin/sh
cd /media/data3/jian/Text_Detection/YOLOV3
rm train_clean.txt valid_clean.txt
mv train.txt train_clean.txt
mv valid.txt valid_clean.txt

cd images
rm -rf train_clean valid_clean
mv train train_clean
mv valid valid_clean

cd ../labels
rm -rf train_clean valid_clean
mv train train_clean
mv valid valid_clean

cd ..
CUDA_VISIBLE_DEVICES=1,2 python3 yolo_split.py --COCO_path /media/data3/jian/Text_Detection/COCO-Text/COCO_Text.json --img_path /media/data3/jian/Text_Detection/COCO-Text/train2014/ --task 4
echo "Job completed!"
