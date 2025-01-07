import torch
import re
import time
from argparse import Namespace
from typing import Dict, List, Tuple
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import AutoImageProcessor
from contextlib import contextmanager
import logging

from utils import load_data, load_pretrained_model, evaluate

@dataclass
class TrainingConfig:
    initial_backbone_lr: float = 1e-6
    initial_encoder_decoder_lr: float = 1e-6
    target_backbone_lr: float = 1e-5
    target_encoder_decoder_lr: float = 1e-4
    max_norm: float = 0.1
    threshold: float = 0.1
    num_epochs: int = 72
    warmup_steps: int = 1000
    validation_interval: int = 2
    batch_size: int=8
    num_workers: int=4

class MemoryManager:
    @staticmethod
    @contextmanager
    def autocast(device: str):
        try:
            with torch.autocast(device_type=device):
                yield
        finally:
            torch.cuda.empty_cache() if device == "cuda" else None

    @staticmethod
    def clear_cache():
        torch.cuda.empty_cache()

class OptimizationManager:
    def __init__(self, model: torch.nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.optimizer, self.scheduler = self._setup_optimizer()
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        
    def _setup_optimizer(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        param_groups = self._create_param_groups()
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.target_encoder_decoder_lr,
            betas=(0.9, 0.999),
            weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[1000], gamma=0.1
        )
        return optimizer, scheduler

    def _create_param_groups(self) -> List[Dict]:
        backbone_pattern = re.compile(r'^(?=.*backbone)(?!.*norm).*$')
        norm_pattern = re.compile(r'^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$')
        
        param_groups = []
        for name, param in self.model.named_parameters():
            if backbone_pattern.match(name):
                param_groups.append({
                    'params': param,
                    'lr': self.config.initial_backbone_lr,
                    'target_lr': self.config.target_backbone_lr
                })
            elif 'encoder' in name or 'decoder' in name:
                param_groups.append({
                    'params': param,
                    'lr': self.config.initial_encoder_decoder_lr,
                    'target_lr': self.config.target_encoder_decoder_lr,
                    'weight_decay': 0.0 if norm_pattern.match(name) else 0.0001
                })
        return param_groups

    def adjust_learning_rate(self, step: int):
        if step < self.config.warmup_steps:
            for group in self.optimizer.param_groups:
                progress = step / self.config.warmup_steps
                group['lr'] = group['target_lr'] * progress
        else:
            self.scheduler.step()

class Trainer:
    def __init__(self, model, config: TrainingConfig, device: str):
        self.model = model
        self.config = config
        self.device = device
        self.opt_manager = OptimizationManager(model, config)
        self.best_metrics = {'map50': 0}
        self.logger = logging.getLogger(__name__)
        
    def train(self, train_loader, val_loader, processor, criterion) -> Dict:
        try:
            step = 0
            for epoch in range(self.config.num_epochs):
                train_loss = self._train_epoch(train_loader, criterion, step)
                step += len(train_loader)
                
                if (epoch + 1) % self.config.validation_interval == 0:
                    self._validate_and_save(val_loader, processor)
                
                self.logger.info(f"Epoch {epoch + 1}: Loss = {train_loss:.4f}\n")
                
            return {
                'best_model': self.best_metrics['model_state'],
                'best_map50': self.best_metrics['map50']
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def _train_epoch(self, train_loader, criterion, global_step: int) -> float:
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc="Training") as progress_bar:
            for batch in progress_bar:
                loss = self._process_batch(batch, criterion, global_step)
                total_loss += loss
                global_step += 1
                
                progress_bar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'lr': f"{self.opt_manager.optimizer.param_groups[0]['lr']:.2e}"
                })
                
        return total_loss / len(train_loader)

    def _process_batch(self, batch, criterion, step: int) -> float:
        with MemoryManager.autocast(self.device):
            try:
                images = batch["pixel_values"].to(self.device, non_blocking=True)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
                
                self.opt_manager.optimizer.zero_grad()
                outputs = self.model(images, targets)
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict.values())
                
                self.opt_manager.scaler.scale(loss).backward()
                
                if step > self.config.warmup_steps:
                    self.opt_manager.scaler.unscale_(self.opt_manager.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_norm
                    )
                
                self.opt_manager.scaler.step(self.opt_manager.optimizer)
                self.opt_manager.scaler.update()
                self.opt_manager.adjust_learning_rate(step)
                
                return loss.item()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    MemoryManager.clear_cache()
                    self.logger.error("OOM error, clearing cache and skipping batch")
                    return 0.0
                raise

    @torch.no_grad()
    def _validate_and_save(self, val_loader, processor):
        self.model.eval()
        map50, map50_95 = evaluate(
            self.model, 
            val_loader,
            processor,
            self.config.threshold,
            self.device
        )
        
        if map50 > self.best_metrics['map50']:
            self.best_metrics.update({
                'map50': map50,
                'model_state': {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            })
            torch.save(self.best_metrics['model_state'], "best.pth")
            self.logger.info(f"New best model saved with mAP50: {map50:.4f}")

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        args = Namespace(
            config_path='configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
            resume_path='models/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth',
            json_train="dataset/visdrone/annotations/train_coco.json",
            json_val="dataset/visdrone/annotations/val_coco.json",
            device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        
        config = TrainingConfig()
        processor = AutoImageProcessor.from_pretrained(
            "PekingU/rtdetr_r18vd_coco_o365",
            do_resize=True,
            size={"width": 640, "height": 640}
        )
        
        train_loader, val_loader = load_data(
            json_train=args.json_train,
            json_val=args.json_val
        )
        
        model, cfg, _ = load_pretrained_model(args.config_path, args.resume_path)
        model.to(args.device)
        
        trainer = Trainer(model, config, args.device)
        start_time = time.time()
        
        results = trainer.train(train_loader, val_loader, processor, cfg.criterion)
        
        logger.info(
            f"Training completed in {(time.time()-start_time)/60:.2f} minutes. "
            f"Best mAP50: {results['best_map50']:.4f}"
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()