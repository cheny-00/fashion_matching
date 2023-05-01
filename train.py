
import torch 
import numpy as np

from tqdm import tqdm
from time import time
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from losses import SimpleTripletLoss, ContrastiveLoss
from triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from center_loss import CenterLoss

class BaseTrainer:
    
    def __init__(self,
                 model,
                 device,
                 optims,
                 params) -> None:
        
        self.model = model
        self.device = device
        self.params = params
        self.optims = optims
        self.device = device

        self.n_epoch = 0
        self.train_step = 0
        self.losses = defaultdict(int)
        self.scheduler = None
        self.warmup_lr = None
        
        self.init_losses()
        self.init_loss_fn()
    
    
    def init_tensorboard(self, path):
        self.tb_writer = SummaryWriter(log_dir=path)
        self.tb_writer.flush()
        
    def init_losses(self,):
        self.losses = defaultdict(int)
    
    def write_tb(self, ):
        for k, v in self.losses.items():
            self.tb_writer.add_scalar(k, v, self.train_step)
    
    def train(self, train_iter):
        self.model.train()
        self.model.to(self.device)
        log_interval = self.params['log_interval']
        tqdm_train_data_loader = tqdm(train_iter, desc='train process', total=len(train_iter))
        start_time = time()
        
        for data in tqdm_train_data_loader:
            self.train_step += 1
            if self.params['do_warmup'] and self.warmup_lr is not None and self.n_epoch <= self.params['n_warmup_step']:
                self.warmup_lr()

            losses = self.train_process(data)
            
            self._grad_update(losses)
            if self.scheduler is not None and self.n_epoch > self.params['n_warmup_step']:
                self.scheduler.step()
            for k, v in losses.items():
                self.losses[k] += v.item()
            
            if self.train_step % log_interval == 0:
                for k, v in self.losses.items():
                    self.losses[k] = v / log_interval
                elapsed, start_time = time() - start_time, time()
                desc_text = self.get_desc(elapsed)
                
                self.write_tb()
                tqdm_train_data_loader.set_description(desc=desc_text, refresh=True)
                self.init_losses()
                
    def init_loss_fn(self, *args, **kwargs):
        raise NotImplementedError
    
    def train_process(self, *args, **kargs):
        raise NotImplementedError
    
    
    def _grad_update(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_desc(self, elapsed):
        return self._get_desc(elapsed)
    
    def _get_desc(self, elapsed):
        epoch_info = f"Epoch:{self.n_epoch}"
        step_info = f"Step:{self.train_step}"
        losses_info = "|".join([f"{k}:{v:.3f}" for k, v in self.losses.items()])
        elapsed_info = f"elapesd:{elapsed:.2f}"
        return "|".join([epoch_info, step_info, losses_info, elapsed_info])
        
    

class TripletEmbedTrainer(BaseTrainer):
    
    def __init__(self, model, device, optims, params) -> None:
        super().__init__(model, device, optims, params)
        self.warmup_lr = self._warmup_lr
    
    
    def _warmup_lr(self, ):
        lr_scale = min(
            1.0,
            float(self.n_epoch) / float(self.params['n_warmup_step'])
        )
        for pg in self.optims[0].param_groups:
            pg['lr'] = lr_scale * self.params['base_lr']
        
    
    def init_loss_fn(self, ):
        self.contrastive_loss_fn = TripletLoss(self.params['margin'], self.params['dist_fn'])
        self.xent_fn = CrossEntropyLabelSmooth(self.params['num_classes'])
    
    def train_process(self, data):
        device = self.device

        x, class_labels, idx, is_real = data['x'].to(device), data['label'].to(device), data['idx'].to(device), data['is_real'].to(device)
        

        features, cls_score = self.model(x)
        
        # embedding loss
        contrastive_loss, dist_ap, dist_an = self.contrastive_loss_fn(features, class_labels, mask=is_real)
        contrastive_loss = contrastive_loss * self.params['query_contrastive_weight']
        
        center_loss = self.model.center_loss_fn(features, class_labels)
        center_loss = self.params['query_center_weight'] * center_loss
        
        
        # classifier loss
        xent_loss = self.xent_fn(cls_score, class_labels)
        xent_loss = xent_loss * self.params['query_xent_weight']
        
        total_loss = contrastive_loss + center_loss + xent_loss
        losses ={
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "center_loss": center_loss,
            "xent_loss": xent_loss
        }
        return losses

    def _grad_update(self, losses):
        
        # implement gradient accumulate
        
        for optim in self.optims:
            optim.zero_grad()
        
        losses['total_loss'].backward()
        self.optims[0].step()
        for param in self.model.center_loss_fn.parameters():
            param.grad.data *= 1.0 / self.params['query_center_weight']
        self.optims[1].step()
        
        # del losses['total_loss']
        
    def get_desc(self, elapsed):
        desc = self._get_desc(elapsed)
        lr_info = self.optims[0].param_groups[0]['lr']
        desc = f"lr:{lr_info:.7f}|" + desc
        return desc