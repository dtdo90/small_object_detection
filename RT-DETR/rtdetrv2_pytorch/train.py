# load model
import torch
import math
from argparse import Namespace
from src.core import YAMLConfig
import numpy as np
import supervision as sv


import albumentations as A
from data_visdrone import VisDroneData
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import AutoImageProcessor

args = Namespace(config_path='configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
                 resume_path='models/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth',
                 json_train="dataset/visdrone/annotations/train_coco.json",
                 json_val="dataset/visdrone/annotations/val_coco.json",
                 summary_dir=None, test_only=False, update=None, print_method='builtin',
                 print_rank=0, local_rank=None)


def load_pretrained_model(config_path,resume_path):

    # initialize the raw model
    cfg=YAMLConfig(config_path, resume=resume_path)
    model=cfg.model
    # model state_dict
    state_dict_model=model.state_dict()

    # pretrained state_dict
    checkpoint=torch.load(args.resume_path,map_location="cpu")
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
                print(f"Matched and loaded weight for: {model_key}")
            else:
                print(f"Shape mismatch for {model_key}: {state_weight.shape} vs {model_param.shape}")

    # Load the matched weights into the model
    model.load_state_dict(matched_weights, strict=False)
    print(f"\nLoad pretrained weights succesfully | {sum(p.numel() for p in model.parameters())/1e6} million parameters")
    return model, cfg

#model, cfg=load_pretrained_model(args.config_path,args.resume_path)

def load_data(json_train, json_val):
    # define colate function
    def collate_fn(batch):
        # Extract pixel values and labels
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        # Prepare labels
        labels = [x["labels"] for x in batch]
        return {"pixel_values": pixel_values, "labels": labels}
    
    # define train and validation transformations
    train_transform = A.Compose(
        [A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1,
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
    train_loader=DataLoader(ds_train,
                            batch_size=8,
                            collate_fn=collate_fn,
                            num_workers=2,
                            shuffle=True,
                            pin_memory=True)
    # load validation data and get a data loader
    ds_val = VisDroneData(
            json_path=json_val,
            split="val",
            transforms=val_transform)
    val_loader=DataLoader(ds_val,
                        batch_size=8,
                        collate_fn=collate_fn,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True)
    print("Create dataloaders succesfully!")
    return train_loader, val_loader


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

# train with linear learning rate warmup and cosine annealing lr scheduler
def train_and_evaluate(
    model, num_epochs, train_loader, val_loader, optimizer, criterion, max_norm, device, 
    processor, threshold, # for evaluation
    warmup_steps, initial_lr=3e-05, min_lr=1e-6, # start with initial_lr till reaching optimizer lr
    amp=True, # allow mixed precision for faster training
    scaler=GradScaler(enabled=True) # use GradScaler in conjuction with amp
    ):
    # return best model state dict and best map50 for validation set
    best_model_state_dict = None
    best_map50 = 0
    # track running losses, map50, learning rates 
    train_losses, val_map50s, track_lrs=[], [], []
    global_step=-1
    # extract learning rate from optimizer
    peak_lr=optimizer.param_groups[0]["lr"]

    # total number of iterations
    num_batches=len(train_loader)
    total_train_steps=num_batches*num_epochs
    # learning rate increment during warmup phase
    lr_increment=(peak_lr-initial_lr)/warmup_steps

    for epoch in range(num_epochs):
        model.train()
        loss_epoch=0

        # tqdm progress bar
        progress_bar=tqdm(train_loader,desc="Training",leave =True)
        for batch_idx, batch in enumerate(progress_bar):
            global_step+=1

            # adjust lr based on current phase
            if global_step<warmup_steps: # warmup
                lr=initial_lr+lr_increment*global_step
            else: # cosine annealing gradually decrease lr till reaching min lr
                progress= (global_step-warmup_steps)/(total_train_steps-warmup_steps)
                lr=min_lr+(peak_lr-min_lr)*0.5*(1+math.cos(math.pi*progress))
            # apply the calculated lr to the optimizer
            for param_group in optimizer.param_groups:
                param_group["lr"]=lr
            track_lrs.append(lr) # store the current lr

            # extract input from batch and move to correct device
            batch_images = batch["pixel_values"].to(device)
            batch_images = batch_images.to(device=device, dtype=torch.float32, non_blocking=True)
            batch_targets = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            # forward with amp (allowed mixed precision)
            with torch.autocast(device_type=device, cache_enabled=True):
                outputs = model(batch_images, batch_targets)
            with torch.autocast(device_type=device, cache_enabled=False):
                loss_dict = criterion(outputs, batch_targets)
            # compute loss and backward
            loss=sum(loss_dict.values())
            scaler.scale(loss).backward() # use scaler in conjunction with amp

            # gradient clipping
            if global_step>warmup_steps and max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            # update parameters
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() # zero grad for the next run


            # add loss of the current batch
            loss_epoch += loss.item()
            # Update tqdm bar with the loss of current batch
            progress_bar.set_postfix({"batch_loss": loss.item()})

        # Close tqdm bar
        progress_bar.close()
        loss_epoch=loss_epoch / num_batches if num_batches > 0
        train_losses.append(loss_epoch) # add epoch loss to the list

        # Evaluate on validation set
        map50, map50_95 = evaluate(
            model,
            val_loader,
            processor=processor,
            threshold=threshold,
            device=device
        )
        val_map50s.append(map50) # add map50 to the list
        # Update best model
        if map50 > best_map50:
            best_map50 = map50
            best_model_state_dict = model.state_dict()  # Save model state dict, not the entire model

        # print out
        print(f"--------- Epoch {epoch + 1}/{num_epochs} --------- ")
        print(f"Train_loss: {loss_epoch:.4f} | val_map50: {map50:.4f} | val_map50_95: {map50_95:.4f}\n")
    
    return best_model_state_dict, best_map50

def main():
    # load pretrained model and cfg 
    model, cfg=load_pretrained_model(args.config_path,args.resume_path)
    # move model to device
    device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 
    model.to(device) # move model to device

    num_epochs=2 
    train_loader, val_loader= load_data(json_train=args.json_train, json_val=args.json_val)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0005) 
    criterion=cfg.criterion 
    max_norm=0.1 
    processor=AutoImageProcessor.from_pretrained(
            "PekingU/rtdetr_r18vd_coco_o365",
            do_resize=True,
            size={"width": 640, "height": 640},)
    threshold=0.01
    warmup_steps=0.2*len(train_loader)*num_epochs # 20% warmup
    # train
    best_model_state_dict, best_map50=train_and_evaluate(
    model, num_epochs, train_loader, val_loader, optimizer, criterion, max_norm, device, 
    processor, threshold, warmup_steps)
    
    # save model dictionary and optimizer
    print("Training complete! Saving the best model ...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
    )

    print(f"Model saved! Best map50: {best_map50}")

if __name__=="__main__":
    main()