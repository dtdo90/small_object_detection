import torch
from data_visdrone import VisDroneData
import albumentations as A
from src.core import YAMLConfig
import supervision as sv
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from argparse import Namespace
from dataclasses import dataclass

from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def load_data(json_train, json_val, test_mode=False):
    # define colate function
    def collate_fn(batch):
        # Extract pixel values and labels
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        # Prepare labels
        labels = [x["labels"] for x in batch]
        return {"pixel_values": pixel_values, "labels": labels}
    
    # define train and validation transformations
    train_transform = A.Compose(
        [A.ShiftScaleRotate(shift_limit=0.1,
                           scale_limit=0.5,
                            rotate_limit=0,
                            p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=70, val_shift_limit=40, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),],
        bbox_params=A.BboxParams(
            format="pascal_voc",  # Albumentations expects [xmin, ymin, xmax, ymax]
            label_fields=["category"],
            clip=True,
            min_area=1,
        ),
    )

    val_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category"],
            clip=True,
            min_area=1,
        ),
    )

    # load train data and get a data loader
    ds_train = VisDroneData(
        json_path=json_train,
        split="train",
        transforms=train_transform)
    # load validation data and get a data loader
    ds_val = VisDroneData(
            json_path=json_val,
            split="val",
            transforms=val_transform)
    # test mode
    train_loader=DataLoader(ds_train,
                            batch_size=8,
                            collate_fn=collate_fn,
                            num_workers=2,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)
    val_loader=DataLoader(ds_val,
                        batch_size=8,
                        collate_fn=collate_fn,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)
    if test_mode:  # Only return the first batch
        train_loader = [next(iter(train_loader))]
        val_loader = [next(iter(val_loader))]
    print("Create dataloaders succesfully!")
    return train_loader, val_loader



def load_pretrained_model(config_path,resume_path):

    # initialize the raw model
    cfg=YAMLConfig(config_path, resume=resume_path)
    model=cfg.model
    # model state_dict
    state_dict_model=model.state_dict()

    # pretrained state_dict
    checkpoint=torch.load(resume_path,map_location="cpu")
    if 'ema' in checkpoint:
        state_dict_pretrained=checkpoint['ema']['module']
    else:
        state_dict_pretrained=checkpoint['model']

    # Create a new state dictionary to store matched weights
    matched_weights = {}

    # Loop through all layers in the model
    for model_key, model_param in state_dict_model.items():
        # Try to find a matching key in state_dict_pretrained
        matched_key = None
        for state_key in state_dict_pretrained.keys():
            # Check if the state_dict key is a substring of the model key
            if state_key in model_key:
                matched_key = state_key
                break

        # If a matching key is found and shapes match, load the weight
        if matched_key is not None:
            state_weight = state_dict_pretrained[matched_key]
            # Ensure the shapes match exactly
            if state_weight.shape == model_param.shape:
                matched_weights[model_key] = state_weight
                #print(f"Matched and loaded weight for: {model_key}")
            # else:
            #     print(f"Shape mismatch for {model_key}: {state_weight.shape} vs {model_param.shape}")

    # Load the matched weights into the model
    model.load_state_dict(matched_weights, strict=False)

    # Disable gradient computation for matched parameters
    # for name, param in model.named_parameters():
    #     if name in matched_weights:
    #         param.requires_grad = False

    print(f"\nLoad pretrained weights successfully | "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters") 
    
    return model, cfg, matched_weights



@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


# compute mAP50 and mAP50-100 in validation
def evaluate(model, loader, processor, threshold, device):
    model.eval()

    # Initialize tqdm progress bar and evaluator
    progress_bar = tqdm(loader, desc="Validating", leave=True)
    evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    evaluator.warn_on_many_detections = False

    for batch in progress_bar:
        # Move batch data to the correct device
        images = batch['pixel_values'].to(device)
        batch_targets = batch['labels']

        # (1) Prepare target sizes and targets
        target_sizes = torch.tensor(np.array([x["orig_size"] for x in batch_targets])).to(device)
        batch_targets_processed = []

        # loop through individual targets
        for target, (height,width) in zip(batch_targets,target_sizes):
            boxes=target['boxes'].cpu().numpy()
            # convert to xyxy and compute actual dimensions
            boxes=sv.xcycwh_to_xyxy(boxes)
            boxes=boxes*np.array([width.item(),height.item(),width.item(),height.item()])
            boxes=torch.tensor(boxes, device=device)
            labels=target["labels"].to(device)
            batch_targets_processed.append({
                "boxes": boxes,
                "labels": labels
            })

        # (2) Compute predictions and post-process them
        with torch.no_grad():
            preds = model(images)
            outputs = ModelOutput(
                logits=preds['pred_logits'],
                pred_boxes=preds['pred_boxes']
            )
            batch_preds_processed = processor.post_process_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=target_sizes
            )

        # (3) Update evaluator incrementally
        preds_for_evaluator = [
            {
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
                "labels": pred["labels"].cpu()
            }
            for pred in batch_preds_processed
        ]
        targets_for_evaluator = [
            {
                "boxes": target["boxes"].cpu(),
                "labels": target["labels"].cpu()
            }
            for target in batch_targets_processed
        ]
        evaluator.update(preds=preds_for_evaluator, target=targets_for_evaluator)

    # Compute final metrics
    print("Computing map ...")
    metrics = evaluator.compute()
    mAP50 = metrics["map_50"].item()
    mAP50_95 = metrics["map"].item()

    #print(f"mAP@50: {mAP50:.4f}, mAP@50-95: {mAP50_95:.4f}")
    return mAP50, mAP50_95


if __name__=="__main__":
    args = Namespace(config_path='configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
                    resume_path='models/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth',
                    json_train="dataset/visdrone/annotations/train_coco.json",
                    json_val="dataset/visdrone/annotations/val_coco.json",
                    device="cuda" if torch.cuda.is_available else "mps" if torch.backends.mps.is_available else "cpu"
                    )
        
    processor=AutoImageProcessor.from_pretrained(
            "PekingU/rtdetr_r18vd_coco_o365",
            do_resize=True,
            size={"width": 640, "height": 640},)
    device=args.device
    # data
    train_loader, val_loader= load_data(json_train=args.json_train, json_val=args.json_val, test_mode=True)
    num_train_batches= len(train_loader)
    num_val_batches=len(val_loader)
    print(f"Data loaded | Train batches: {num_train_batches} | Val batches: {num_val_batches}")
    
    for batch in train_loader:
        batch_images = batch["pixel_values"]
        batch_targets=batch["labels"]
        print(batch_images.shape)
        print(len(batch_targets))
        break

    model, cfg, matched_weights=load_pretrained_model(config_path=args.config_path,resume_path=args.resume_path)
    num_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded successfully! Trainable parameters: {num_params:,}")
