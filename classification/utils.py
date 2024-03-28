import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from torchvision import transforms as tf
import PIL
import logging

# copied from https://github.com/huggingface/diffusers/blob/cb9d77af23f7e84fb684c7c87b3de35247ba1d8b/src/diffusers/utils/torch_utils.py#L29-L70
def randn_tensor(
    shape: Union[Tuple, List],
    # logger,
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
            # if device != "mps":
            #     logger.info(
            #         f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
            #         f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
            #         f" slighly speed up this function by passing a generator that was created on the {device} device."
            #     )
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


def get_depth_pil(depth_path):

    depth_pil = Image.open(depth_path)
    depth_image = (np.clip(np.array(depth_pil).astype(np.float64) / 8000, 0., 1.) * 255).astype(np.uint8)
    if depth_image.ndim == 2:
        depth_image = depth_image[:,:,np.newaxis]
        depth_image = np.concatenate([depth_image, depth_image, depth_image], axis=2)
    depth_pil_scaled = Image.fromarray(depth_image)

    depth_image_inv = 255 - depth_image
    if depth_image_inv.ndim == 2:
        depth_image_inv = depth_image_inv[:,:,np.newaxis]
        depth_image_inv = np.concatenate([depth_image_inv, depth_image_inv, depth_image_inv], axis=2)
    depth_pil_inv_scaled = Image.fromarray(depth_image_inv)

    return depth_pil_scaled, depth_pil_inv_scaled

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
        if img.size(0) == 1:
            img = img.repeat(3,1,1)
        batch = torch.cat([batch,img.unsqueeze(0)],dim=0)

    return batch


# https://github.com/huggingface/diffusers/blob/7a24977ce3f7b406034362c15c17b4159abe7dfd/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py#L38C1-L152C30
def prepare_mask_and_masked_image(image, mask, height, width, return_image: bool = False):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """

    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image


def add_new_tokens(tokenizer, text_encoder, template, num_new_tokens, init_mode, embs_path=None, init_token=None):
    # Add the placeholder tokens in tokenizer
    new_tokens = [f'<{template}_{i}>' for i in range(num_new_tokens)]
    num_added_tokens = tokenizer.add_tokens(new_tokens)

    if num_added_tokens != num_new_tokens:
        raise ValueError('Tokens are added incorrectly.')

    logging.info(f"Added {num_added_tokens} tokens to the tokenizer.")

    new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Convert the initializer_token, placeholder_token to ids

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data

    if init_mode == 'mean_emb_cov_emb':
        mean = token_embeds.cpu().numpy().mean(axis=0)
        cov = np.cov(token_embeds.cpu().numpy(), rowvar=0)
        weights = np.random.multivariate_normal(mean=mean, cov=cov, size=num_new_tokens)
        for new_token_id, weight in zip(new_token_ids, weights):
            token_embeds[new_token_id] = torch.tensor(weight)
    elif init_mode == 'init_token':
        token_ids = tokenizer.encode(init_token, add_special_tokens=False)
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
        for new_token_id in new_token_ids:
            token_embeds[new_token_id] = token_embeds[initializer_token_id]
    elif init_mode == 'file':
        if embs_path is None:
            raise ValueError("The path to the embeddings file must be specified.")

        embeddings_dict = torch.load(embs_path)
        embeddings = [embeddings_dict[token] for token in new_tokens]

        for placeholder_token, embedding in zip(new_tokens, embeddings):
            placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
            token_embeds[placeholder_token_id] = embedding.clone().detach()
    else:
        raise ValueError(f"Invalid init_mode: {init_mode}")

    return new_token_ids, new_tokens


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