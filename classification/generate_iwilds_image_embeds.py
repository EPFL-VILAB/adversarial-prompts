import torch

import matplotlib.pyplot as plt
import numpy as np
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torch
import pandas as pd

import sys
from transformers import CLIPVisionModel, CLIPImageProcessor



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.bfloat16

torch.set_grad_enabled(False)


import torchvision.transforms as tf
import PIL

def preprocess_clip(images, resolution=224):

    if isinstance(images, list):
        if resolution is not None:
            images = [image.resize((resolution,resolution), resample=PIL.Image.BILINEAR) for image in images]
        images = torch.stack([tf.ToTensor()(i) for i in images], dim=0)

    if resolution is not None:
        resize_transform = tf.Resize(size=(resolution, resolution), antialias=True)
        images = resize_transform(images)

    ## obtained from mask2former_preprocessor
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    device = images.device
    image_mean = image_mean.to(device)
    image_std = image_std.to(device)
 
    normalized_tensor = (images  - image_mean[None, :, None, None]) / image_std[None, :, None, None]

    return normalized_tensor




pretrained_model_name_or_path= "openai/clip-vit-large-patch14"
perceptual_loss_model = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path)
perceptual_loss_model.eval()
image_processor = CLIPImageProcessor.from_pretrained(pretrained_model_name_or_path)


from data.iwilds import WILDS
import torch
batch_size = 16
num_images =64
np.random.seed(42)

for location in [287,288]:
    for i in range(2):
        if i == 0:
            test_set = WILDS(split="test", night_time=True, keep_location=[location], transform=tf.ToTensor())
        else:
            test_set =  WILDS(split="test", daylight_time=True, keep_location=[location],  transform=tf.ToTensor())


        if num_images is not None:

            indicies = np.random.choice(len(test_set), num_images, replace=False)
            test_set = torch.utils.data.Subset(test_set, indicies)


        print(len(test_set))
        dataloader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False
        )
        tot_features = torch.empty(0,1024)
        with torch.no_grad():
            for images, y, locations in dataloader:
                images = preprocess_clip(images)
                features = perceptual_loss_model(images).pooler_output
                tot_features = torch.cat([tot_features, features])

        mean_features = tot_features.mean(dim=0)
        std_features = tot_features.std(dim=0)
        print(mean_features.shape)

        time = "night" if i == 0 else "day"
        save_name = f"test_{time}_loc={location}_clip_embeds.pt"
        save_path = f"target_image_embeds/{save_name}"

        torch.save({"mean":mean_features, "std": std_features}, save_path)

        #torch.save(mean_features, save_path)


