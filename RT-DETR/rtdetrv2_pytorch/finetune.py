import torch
import time
from argparse import Namespace

from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import AutoImageProcessor
from utils import load_data, load_pretrained_model, evaluate,calc_loss_batch,calc_loss_loader

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# pip install pycocotools supervision torchmetrics albumentations pycocotools --upgrade transformers


def choose_optimizer(model,matched_weights,warmup=False):
    """ Choose optimizer based on the warmup state"""
    import re
    if warmup: # Optional: to train the new parameters for a few epochs 
        # disable all gradients in the pretrained weights
        for name, param in model.named_parameters():
            if name in matched_weights:
                param.requires_grad = False
        return torch.optim.AdamW(model.parameters(), lr=4e-3)
    # Configuration from original paper: different lr and scheduler for backbone vs encodeer_decoder
    else: 
        # Define regex patterns
        backbone_pattern = re.compile(r'^(?=.*backbone)(?!.*norm).*$')
        norm_pattern = re.compile(r'^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$')
        
        # Parameter groups setup
        backbone_params = []
        encoder_decoder_params = []
        
        # Separate parameters based on regex patterns
        for name, param in model.named_parameters():
            if backbone_pattern.match(name):
                backbone_params.append({
                    'params': param,
                    'lr': 1e-5
                })
            elif 'encoder' in name or 'decoder' in name:
                if norm_pattern.match(name):
                    encoder_decoder_params.append({
                        'params': param,
                        'weight_decay': 0.0
                    })
                else:
                    encoder_decoder_params.append({
                        'params': param,
                    })
        
        # Create optimizers
        backbone_optimizer = torch.optim.AdamW(
            backbone_params,
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.0001
        )
        
        encoder_decoder_optimizer = torch.optim.AdamW(
            encoder_decoder_params,
            lr=1e-4,
            betas=(0.9, 0.999),
            weight_decay=0.0001
        )
        
        # Create step-based schedulers
        backbone_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            backbone_optimizer,
            milestones=[1000],  # Will reduce lr at step 1000
            gamma=0.1
        )
        
        encoder_decoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            encoder_decoder_optimizer,
            milestones=[1000],  # Will reduce lr at step 1000
            gamma=0.1
        )
        return backbone_optimizer, backbone_scheduler, encoder_decoder_optimizer, encoder_decoder_scheduler


def train_and_evaluate(
    args, num_epochs, # num_epochs_warmup trains only the new params
    train_loader, val_loader, max_norm, device, 
    processor, threshold, # for evaluation
    warmup_duration=2000,  # learning rate warmup
    scaler=GradScaler(enabled=True) # use gradscaler in conjuction with amp
    ):
    """ Train with linear lr warmup and lr scheduler"""
    
    # load pretrained model and move to device
    model, cfg, matched_weights=load_pretrained_model(args.config_path,args.resume_path)
    model.to(device) 
    criterion=cfg.criterion

    # return best model state dict and best map50 for validation set
    best_model_state_dict = None
    best_map50 = 0
    # track running losses, map50, learning rates 
    train_losses, val_map50s, track_lrs=[], [], []
    global_step=-1

    # unfreeze all weights
    backbone_optimizer, backbone_scheduler, encoder_decoder_optimizer, encoder_decoder_scheduler=choose_optimizer(
        model=model,matched_weights=matched_weights,warmup=False
    )

    # compute initial map metrics
    map50, map50_95 = evaluate(model,val_loader,processor,threshold,device)
    print(f"Initial map50 : {map50}\n")
    torch.cuda.empty_cache()  # Clear memory after initial evaluation

    # start training
    for epoch in range(num_epochs):
        model.train()
        loss_epoch=0

        # Calculate warmup parameters
        num_batches = len(train_loader)     
        # Initial learning rates for warmup phase
        initial_backbone_lr = 1e-6
        initial_encoder_decoder_lr = 1e-6
        # Target learning rates
        target_backbone_lr = 1e-5
        target_encoder_decoder_lr = 1e-4
        
        # Learning rate increments for warmup
        backbone_lr_increment = (target_backbone_lr - initial_backbone_lr) / warmup_duration
        encoder_decoder_lr_increment = (target_encoder_decoder_lr - initial_encoder_decoder_lr) / warmup_duration
        
        progress_bar=tqdm(train_loader,desc="Training",leave =True)  
        for batch_idx, batch in enumerate(progress_bar):
            global_step += 1
            # Warmup phase learning rate adjustment
            if global_step < warmup_duration:
                backbone_lr = initial_backbone_lr + backbone_lr_increment * global_step
                encoder_decoder_lr = initial_encoder_decoder_lr + encoder_decoder_lr_increment * global_step
                
                # Update learning rates for each parameter group
                for param_group in backbone_optimizer.param_groups:
                    param_group["lr"] = backbone_lr
                for param_group in encoder_decoder_optimizer.param_groups:
                    param_group["lr"] = encoder_decoder_lr
            else:
                # Step schedulers after warmup (per step instead of per epoch)
                backbone_scheduler.step()
                encoder_decoder_scheduler.step()
            
            # Track learning rates
            track_lrs.append({
                'backbone': backbone_optimizer.param_groups[0]["lr"],
                'encoder_decoder': encoder_decoder_optimizer.param_groups[0]["lr"]
            })

             # Zero gradients before forward pass
            backbone_optimizer.zero_grad()
            encoder_decoder_optimizer.zero_grad()

            # compute loss and backward
            loss=calc_loss_batch(model,batch,criterion,device)            
            scaler.scale(loss).backward()

            # Gradient clipping
            if global_step > warmup_duration and max_norm > 0:
                scaler.unscale_(backbone_optimizer)
                scaler.unscale_(encoder_decoder_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # Update parameters
            scaler.step(backbone_optimizer)
            scaler.step(encoder_decoder_optimizer)
            scaler.update()

            loss_epoch += loss.item()
            progress_bar.set_postfix({
                "batch_loss": loss.item(),
                "step": global_step,
                #"backbone_lr": f"{backbone_optimizer.param_groups[0]['lr']:.2e}",
                #"encoder_decoder_lr": f"{encoder_decoder_optimizer.param_groups[0]['lr']:.2e}"
            })

        progress_bar.close()

        # Calculate epoch metrics
        loss_epoch = loss_epoch / num_batches 
        train_losses.append(loss_epoch)

        # Validation
        if (epoch+1)%5==0:
            loss_val=calc_loss_loader(model,val_loader,criterion,device)
            map50, map50_95 = evaluate(
                model,
                val_loader,
                processor=processor,
                threshold=threshold,
                device=device
            )
            val_map50s.append(map50)

            # Update and save the best model + optimizer
            if map50 > best_map50:
                best_map50 = map50
                best_model_state_dict = model.state_dict()
                print("Saving the best model ...")
                torch.save(best_model_state_dict,"best.pth")
            # Print detailed training information
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"train_loss: {loss_epoch:.4f} | val_loss: {loss_val:.4f} |val_map50: {map50:.4f} | val_map50_95: {map50_95:.4f}\n")
            print(50*"-")
            # Clear memory after validation
            torch.cuda.empty_cache()
        else:
            # Print detailed training information
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train_loss: {loss_epoch:.4f}\n")
            print(50*"-")
    return best_model_state_dict, best_map50



def main():
    args = Namespace(config_path='configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
                 resume_path='models/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth',
                 json_train="dataset/visdrone/annotations/train_coco.json",
                 json_val="dataset/visdrone/annotations/val_coco.json",
                 device="cuda" if torch.cuda.is_available else "mps" if torch.backends.mps.is_available else "cpu")
     
    processor=AutoImageProcessor.from_pretrained(
            "PekingU/rtdetr_r18vd_coco_o365",
            do_resize=True,
            size={"width": 640, "height": 640},)
    max_norm=0.1
    threshold=0.1
    num_epochs=50
    device=args.device
    # data
    train_loader, val_loader= load_data(json_train=args.json_train, json_val=args.json_val)

    # train
    start_time=time.time()
    torch.cuda.empty_cache() # clear cache before training
    best_model_state_dict, best_map50=train_and_evaluate(
    args, num_epochs, 
    train_loader, val_loader, max_norm, device, processor, threshold)
    
    end_time=time.time()
    
    # save model dictionary and optimizer
    print(f"Training complete in {(end_time-start_time)/60:.2f} minutes | Best map50: {best_map50}")

if __name__=="__main__":
    main()