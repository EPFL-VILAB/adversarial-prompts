import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torchvision.transforms.functional import crop
from torchvision import transforms
from functools import partial

def crop_wilds(image, resolution):
    top = int(10*resolution/448)
    left = 0
    height = int(400*resolution/448)
    width = resolution
    
    return crop(image, top, left, height, width)

class IWildsAdversarialLoss:
    def __init__(self,resolution, type='background') -> None:
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
        self.type = type
        self.iwildcam_train_transform = transforms.Compose([
                                #   transforms.Grayscale(num_output_channels=3),
                                  transforms.Lambda(partial(crop_wilds, resolution=resolution)),
                                  transforms.Resize((224, 224), antialias=True)])

    def get_loss(self, model, gen_images, batch):

        gen_images = self.iwildcam_train_transform(gen_images)
        preds = model(gen_images)

        # target class is background!
        if self.type == 'background':
            target = torch.ones(gen_images.shape[0], dtype=torch.long).to(gen_images.device) * 0
            loss = self.loss_fn(preds, target)
        elif self.type == 'uniform':
            label = batch["labels"]

            log_preds = torch.log_softmax(preds, dim=-1)

            # ignore target class
            weights = torch.ones_like(log_preds)
            #weights[:, label] = 0
            weights = weights / weights.sum(1, keepdim=True)

            loss = -(log_preds * weights).sum(dim=-1).mean()
            loss -= np.log(preds.shape[1])
        elif self.type == "cross_entropy":
            label=batch["labels"]
            loss = - F.cross_entropy(preds, label)
        else:
            raise NotImplementedError(f"Unknown type {self.type}")


        return loss
    
    def get_eval_metrics(self, model, gen_images, batch):
        bs = gen_images.shape[0]
        device = gen_images.device
        weight_dtype = gen_images.dtype
        labels = batch["labels"]

        target_background = torch.ones(bs, dtype=torch.long).to(device) * 0
        #target_giraffe = torch.ones(bs, dtype=torch.long).to(device) * 49
        true_target = labels

        orig_images = 0.5*(1.+batch["pixel_values"])
        orig_images = self.iwildcam_train_transform(orig_images) 

        preds_orig = model(orig_images.to(dtype=weight_dtype))
        loss_orig = self.loss_fn(preds_orig,true_target)
        acc_orig = (preds_orig.argmax(dim=-1) == true_target).float().mean().item()
        loss_orig_background = self.loss_fn(preds_orig,target_background)
        acc_orig_background = (preds_orig.argmax(dim=-1) == target_background).float().mean().item()

        gen_images = self.iwildcam_train_transform(gen_images)
        preds_adv = model(gen_images.to(dtype=weight_dtype))
    
        loss_adv = self.loss_fn(preds_adv, true_target)
        acc_adv = (preds_adv.argmax(dim=-1) == true_target).float().mean().item()
    
        loss_adv_background = self.loss_fn(preds_adv, target_background)
        acc_adv_background = (preds_adv.argmax(dim=-1) == target_background).float().mean().item()

        return {
            "loss_target": loss_adv.detach().item(), 
            "acc_target": acc_adv,
            "loss_background": loss_adv_background.item(), 
            "acc_background": acc_adv_background,

            "loss_orig_background": loss_orig_background.item(), 
            "acc_orig_background": acc_orig_background,
            "loss_orig_target": loss_orig.item(), 
            "acc_orig_target": acc_orig,
        }

   
