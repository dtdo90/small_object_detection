
## Models trained with ultralytics framework:
### 1. RT-DETR L:
Download from the following link and put it in under models:
https://drive.google.com/file/d/1v8MLllwiHPgyPFujrn_5q4-jj7nln77P/view?usp=drive_link
### 2. Yolo11-m:
Download from the following link and put it in under models:
https://drive.google.com/file/d/1p8XjyBqHfXXQFjQHhIMbljrS_HNx9lqV/view?usp=sharing

## Custom train RT-DETR v2
1. Clone the original RT-DETR repo from https://github.com/lyuwenyu/RT-DETR.git. We will work with the subfolder rtdetrv2_pytorch
2. Download VisDrone train+ validation images from https://github.com/VisDrone/VisDrone-Dataset.
Then put them under RT-DETR/rtdetrv2_pytorch/dataset/visdrone/train and RT-DETR/rtdetrv2_pytorch/dataset/visdrone/val
3. Get json files in coco format by running **write_json.py**
4. Train with **train.py**
