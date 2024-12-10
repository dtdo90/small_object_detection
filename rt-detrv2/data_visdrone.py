# dataset
import torch, os, yaml,cv2
from PIL import Image
import torchvision
import numpy as np
import json

import albumentations as A

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor

from pycocotools.coco import COCO


class VisDroneData(Dataset):
    def __init__(self,json_path, split, transforms=None):
        # yaml_path= VisDrone yaml file
        super().__init__()

        if split=="train":
            self.root_dir="/content/drive/MyDrive/rt-detrv2-fine-tune/RT-DETR/rtdetrv2_pytorch/dataset/visdrone/train"
        elif split=="val":
            self.root_dir="/content/drive/MyDrive/rt-detrv2-fine-tune/RT-DETR/rtdetrv2_pytorch/dataset/visdrone/val"

        self.transforms=transforms       # transform for fine tuning

        # post process for RTDETR: resize all to 640x640, convert to tensor, normalize image 
        self.post_transform= AutoImageProcessor.from_pretrained(
            "PekingU/rtdetr_r18vd_coco_o365",
            do_resize=True,
            size={"width": 640, "height": 640},)

        with open(json_path,'r') as f:
            self.data_coco=json.load(f)
        
        # image info: each image = dict {"id","file_name", "width","height"}
        self.images_info={image["id"]: image for image in self.data_coco["images"]}
        # dataset indices
        self.image_ids=list(self.images_info.keys())

        # annotation info
        self.annotations={}
        for ann in self.data_coco["annotations"]:
            img_id=ann["image_id"]
            # set annotation at img_id as a list of annotations
            if img_id not in self.annotations:
                self.annotations[img_id]=[]
            self.annotations[img_id].append(ann)

    def __len__(self):
        return len(self.image_ids)

    # Convert COCO bbox to Pascal VOC
    def coco_to_pascal_voc(self,boxes):
        pascal_boxes = []
        for box in boxes:
            x, y, w, h = box
            pascal_boxes.append([x, y, x+w, y+h])
        return torch.tensor(pascal_boxes, dtype=torch.float32)
    
    @staticmethod
    def annotations_formatted(image_id, categories, bboxes):
        """ Convert categories and bboxes to COCO format annotations.
            bbox= [x1,x2,y1,y2] needs to be converted to [x1,y1,w,h]
        """
        annotations_formatted = []
        for category, bbox in zip(categories, bboxes):
            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1
            annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": [x1, y1, width, height],
                "iscrowd": 0,
                "area": width * height,
            }
            annotations_formatted.append(annotation)
        return {"image_id": image_id, "annotations": annotations_formatted}

    # rescale the bouding boxes according to new image sizes
    def scale_boxes(self,boxes, orig_size, new_size):
        orig_h, orig_w = orig_size
        new_h, new_w = new_size
        
        # Scale x, y, width, height
        scaled_boxes = boxes.clone()
        scaled_boxes[:, 0] *= (new_w / orig_w)  # x
        scaled_boxes[:, 1] *= (new_h / orig_h)  # y
        scaled_boxes[:, 2] *= (new_w / orig_w)  # width
        scaled_boxes[:, 3] *= (new_h / orig_h)  # height
        
        return scaled_boxes
    
    def __getitem__(self,idx):
        # get image id
        image_id=self.image_ids[idx]
        
        # image info: {"id","file_name", "width","height"}
        image_info=self.images_info[image_id]
        orig_width, orig_height=image_info["width"], image_info["height"]
        # image in PIL format
        image_path=os.path.join(self.root_dir,image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        # prepare target
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        # read annotations
        for ann in self.annotations[image_id]:
            boxes.append(ann["bbox"]) # bbox in [xmin, ymin, width, height] format
            labels.append(ann["category_id"])
            areas.append(ann["area"])
            iscrowd.append(ann["iscrowd"])

        # convert to tensors
        boxes=torch.tensor(boxes, dtype=torch.float32)
        labels=torch.tensor(labels,dtype=torch.int64)
        areas=torch.tensor(areas,dtype=torch.float32)
        iscrowd=torch.tensor(iscrowd,dtype=torch.int64)
        
        # Convert COCO bbox to Pascal VOC for Albumentations
        boxes_pascal = self.coco_to_pascal_voc(boxes) # [xmin, ymin, xmax, ymax]


        # Apply Albumentations transform
        if self.transforms:
            transformed=self.transforms(
                image=np.array(image),
                bboxes=boxes_pascal.numpy(),
                category=labels.numpy()
            )
            # Check the keys in the transformed dictionary
            # print(transformed.keys())  # Debug: print available keys
            # convert back to tensors
            bboxes=transformed['bboxes']
            categories=transformed['category']
        
        # Process image and annotations
        annotations_formatted=self.annotations_formatted(image_id, categories, bboxes)
    
        processed = self.post_transform(
            images=image,
            annotations=annotations_formatted,
            return_tensors="pt",   
        ) # dict with keys "pixel_values" + "labels", labels = dict {'size', 'image_id', 'class_labels', 'boxes', 'area', 'iscrowd', 'orig_size'}

        # rename the keyworld 'class_labels' to 'labels'
        # Rename 'class_labels' to 'labels' in the nested dictionary
        processed_dict = {k: v[0] for k, v in processed.items()}
        
        if 'labels' in processed_dict.keys() and 'class_labels' in processed_dict['labels']:
            processed_dict['labels']['labels'] = processed_dict['labels'].pop('class_labels')

        return processed_dict

if __name__=="__main__":
    train_transform = A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",  # Albumentations expects [xmin, ymin, xmax, ymax]
            label_fields=["category"],
            clip=True,
            min_area=1,
        ),
    )

    ds_train = VisDroneData(
        json_path="dataset/visdrone/annotations/train_coco.json", 
        split="train", 
        transforms=train_transform)
    print("Number of train data: ",len(ds_train))
    sample=ds_train[0]
    pixel_values=sample['pixel_values']
    target=sample['labels']
    print(f"Pixel values: {pixel_values.shape}")
    print(f"Target type: {type(target)}")
    print(target)