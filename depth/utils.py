import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import transforms as tf
import PIL
import torch.nn.functional as F

import pdb

# copied from https://github.com/huggingface/diffusers/blob/cb9d77af23f7e84fb684c7c87b3de35247ba1d8b/src/diffusers/utils/torch_utils.py#L29-L70
def randn_tensor(
    shape: Union[Tuple, List],
    logger,
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def get_depth_pil(depth_path, dataset_name='taskonomy'):
    depth_pil = Image.open(depth_path)
    
    if dataset_name == 'taskonomy':
        depth_image = (np.clip(np.array(depth_pil).astype(np.float64) / 8000, 0., 1.) * 255) 
    else:
        depth_image = (np.clip(np.array(depth_pil).astype(np.float64) / (2**16-1), 0., 1.) * 255) 

    if depth_image.ndim == 2:
        depth_image = depth_image[:,:,np.newaxis]
        depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)

    mask = depth_image > 254
    inv_gt_norm = (255-(depth_image)).astype(float)
    inv_gt_norm[~mask] = (inv_gt_norm[~mask] - inv_gt_norm[~mask].min()) / (inv_gt_norm[~mask].max() - inv_gt_norm[~mask].min())
    inv_gt_norm[mask] = 0.
    depth_image_inv = (255*inv_gt_norm).astype(np.uint8)
    depth_pil_scaled = Image.fromarray(255 - depth_image_inv)
    depth_pil_inv_scaled = Image.fromarray(depth_image_inv)
        

    return depth_pil_scaled, depth_pil_inv_scaled


def get_canny_pil(depth_path, building, mask_canny):
    mask_data_root = f'/datasets/taskonomymask/{building}_mask_valid/'
    image_mask = Image.open(os.path.join(mask_data_root, depth_path))

    rgb_path = depth_path.replace('depth_zbuffer', 'rgb')
    low_threshold = 100
    high_threshold = 200
    image = Image.open(rgb_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    if mask_canny:
        canny_img = Image.fromarray((image * np.array(image_mask)[..., None]).astype(np.uint8))
    else:
        canny_img = Image.fromarray(image.astype(np.uint8))
    return canny_img
     


def get_normal_pil(depth_path):

    depth_pil = Image.open(depth_path)
    depth_image = 255 - np.array(depth_pil)
    depth_pil_scaled = Image.fromarray(depth_image)

    return depth_pil_scaled


def get_batch_from_pils(pils,resize=256):

    trans = tf.Compose([tf.Resize((resize,resize), interpolation=PIL.Image.BILINEAR),
                            tf.CenterCrop((resize,resize)),
                            tf.ToTensor()])
    batch = torch.Tensor()
    for pil in pils:
        img = trans(pil)
        batch = torch.cat([batch,img.unsqueeze(0)],dim=0)

    return batch

def resize_img(img_tensor,scale_factor=0.65):

    # assume img_tensor is C X H X W
    img_tensor = F.interpolate(img_tensor.unsqueeze(0),scale_factor=(scale_factor,scale_factor))

    return img_tensor

def prepare_images_for_clip(images: Union[torch.Tensor, List[PIL.Image.Image]], resolution=None) -> torch.Tensor:

    if not isinstance(images,(torch.Tensor,list)):
        raise ValueError(f"Wrong type {type(images)}")

    if isinstance(images, list):
        if resolution is not None:
            images = [image.resize((resolution,resolution), resample=PIL.Image.BILINEAR) for image in images]
        images = torch.stack([tf.ToTensor()(i) for i in images], dim=0)

    if resolution is not None:
        resize_transform = tf.Resize(size=(resolution, resolution))
        images = resize_transform(images)

    ## obtained from mask2former_preprocessor
    image_mean = torch.tensor([0.48500001430511475,0.4560000002384186,0.4059999883174896])
    image_std = torch.tensor([0.2290000021457672, 0.2239999920129776,0.22499999403953552])

    device = images.device
    image_mean = image_mean.to(device)
    image_std = image_std.to(device)
 
    normalized_tensor = (images  - image_mean[None, :, None, None]) / image_std[None, :, None, None]

    return normalized_tensor
