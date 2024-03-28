#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import PIL
import json
import logging
import os
import pickle
from timeit import default_timer as timer
from datetime import datetime


import numpy as np

import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import check_min_version

from diffusers import ControlNetModel
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline

# ------------------------------------------------------------------------------

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__)

class ImageDataset(Dataset):
    def __init__(
        self,
        all_images,
        args
    ):
        print(f'total images: {len(all_images)}')

        self.image_paths = all_images
        self.num_images = len(self.image_paths)
        self._length = self.num_images

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}

        if 'taskonomy' in self.image_paths[i]:
            image_depth = Image.open(self.image_paths[i])
            building = self.image_paths[i].split('/')[-3].split('_')[0]
            image_mask = Image.open(os.path.join(args.taskonomy_mask_root, f'{building}_mask_valid/', self.image_paths[i].split('/')[-1]))
        else:
            image_depth = Image.open(self.image_paths[i])
            image_mask = Image.open(self.image_paths[i].replace('depth_zbuffer','mask_valid'))

        if 'hypersim' in self.image_paths[i] or 'blended_mvg' in self.image_paths[i]:
            image_mask = image_mask.resize((512, 384), resample=PIL.Image.BILINEAR)
            image_depth = image_depth.resize((512, 384), resample=PIL.Image.BILINEAR)

        image_depth = np.array(image_depth).astype(np.float32)
        image_depth = image_depth[np.newaxis,:,:]
        image_depth = np.concatenate([image_depth, image_depth, image_depth], axis=0)

        # --- start
        image_depth = torch.from_numpy(np.clip(image_depth / (2**16 - 1), 0., 1.))

        inv_gt_norm = (1 - image_depth).float().numpy()
        
        mask = np.array(image_mask).astype(bool)
        mask = np.repeat(mask[None,...],3,0)

        inv_gt_norm[mask] = (inv_gt_norm[mask] - inv_gt_norm[mask].min()) / (inv_gt_norm[mask].max() - inv_gt_norm[mask].min())
        inv_gt_norm[~mask] = 0.
        depth_image_inv = (255*inv_gt_norm).astype(np.uint8)
        # print(f'TYPE: {depth_image_inv.dtype}; shape: {depth_image_inv.shape}')
        depth_image_inv = torch.from_numpy(depth_image_inv).permute(1,2,0)
        
        example["depth"] = depth_image_inv
        example["depth_path"] = self.image_paths[i]            

        return example

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--images-path", type=str, nargs='+', help="Path to the file containing image paths.",)
    
    parser.add_argument("--random-sample-images", type=int, default=-1, 
                        help="Number of images to sample if sample random imgages. If `-1`, using `--images-path`.",)

    parser.add_argument("--save-root", type=str, default=None, required=True, help="Save root for the dataset",)

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (per device).")

    parser.add_argument("--num-inference-steps", type=int, default=4, help=("Number of inference steps to run during training."),)
    parser.add_argument("--prompt", type=str, required=True, help=("Prompt for generation"),)

    parser.add_argument("--sampling", type=str, default=None, choices=["mean_emb_cov_emb", "mean_room_cov_adv", "mean_adv_cov_adv"],
        help=("Whether to use sampling"),)

    parser.add_argument("--adversarial", action='store_true', help="Whether to do adversarial generation.",)

    parser.add_argument("--adversarial-runs-file", type=str, default=None, help=("File with adversarial paths"),)

    parser.add_argument("--num-new-tokens", type=int, default=None, help="Number of tokens for adversarial generation / sampling.")
    parser.add_argument("--taskonomy-mask-root", type=str, default=None, help="Path to folder containing depth masks for taskonomy.")
    parser.add_argument("--sdedit", action='store_true', help="Whether to do img2img with sdedit")
    parser.add_argument("--strength", type=float, help="Strength for sdedit")

    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--tokens-root", type=str, default=None)

    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1., help='Controlnet conditioning scale')
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    seed = args.seed
    set_seed(seed)

    accelerator_project_config = ProjectConfiguration()

    accelerator = Accelerator(
        mixed_precision='fp16',
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    run_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    # Load tokenizer, scheduler and models
    tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet", torch_dtype=torch.float16)
    noise_scheduler = DDIMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler")
    noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device) 

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)

    if args.sdedit:
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        controlnet=controlnet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        torch_dtype=torch.float16,
    )
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5',
            controlnet=controlnet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            torch_dtype=torch.float16,
        )

    pipe.safety_checker = None
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(accelerator.device)
    pipe.set_progress_bar_config(disable=True)

    # ---- Re-starting logic ----

    data = []
    for path in args.images_path:
        with open(path, 'r') as f:
            data.extend(f.readlines())

    if args.gpu_id is not None:
        start = int(np.floor(len(data) / args.num_gpus * args.gpu_id))
        end = int(np.floor(len(data) / args.num_gpus * (args.gpu_id + 1)))
        data = data[start:end]
    
    data = [x.strip() for x in data]

    # ---- add new tokens if we use sampling strategy ----
    if args.sampling is not None:
        token_embeds = pipe.text_encoder.get_input_embeddings().weight.data
        if args.sampling == 'mean_room_cov_adv' or args.sampling == 'mean_adv_cov_adv':

            with open(args.adversarial_runs_file, 'r') as f:
                tmp = json.load(f)

            adv_embeddings = []
            for path in tmp:
                for epoch in tmp[path]:
                    full_path = os.path.join(args.tokens_root, path, f'learned_embeds-{epoch}.bin')
                    embedding_dict = torch.load(full_path, map_location='cpu')
                    for value in embedding_dict.values():
                        adv_embeddings.append(value)
                    
            if args.sampling == 'mean_adv_cov_adv':
                mean = np.mean(torch.stack(adv_embeddings).numpy(), axis=0)
                cov = np.cov(torch.stack(adv_embeddings).numpy(), rowvar=0)
            elif args.sampling == 'mean_room_cov_adv':
                room_token_id = pipe.tokenizer.convert_tokens_to_ids('room')
                mean = token_embeds[room_token_id].cpu()
                cov = np.cov(torch.stack(adv_embeddings).numpy(), rowvar=0)

        elif args.sampling == 'mean_emb_cov_emb':
            mean = token_embeds.cpu().numpy().mean(axis=0)
            cov = np.cov(token_embeds.cpu().numpy(), rowvar=0)
        
        placeholder_toks = [f'<new_token_{i}>' for i in range(args.num_new_tokens)]
        num_added_tokens = pipe.tokenizer.add_tokens(placeholder_toks)
        if num_added_tokens != len(placeholder_toks):
            raise ValueError('Not all the tokens were added!')

        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        placeholder_token_ids = pipe.tokenizer.convert_tokens_to_ids(placeholder_toks)
        token_embeds = pipe.text_encoder.get_input_embeddings().weight.data

        weights = np.random.multivariate_normal(mean=mean, cov=cov, size=args.num_new_tokens)

        # dump tokens for reproducibility
        os.makedirs(args.save_root, exist_ok=True)
        with open(os.path.join(args.save_root, f'tokens_{run_start_time}_{os.getpid()}.pkl'), 'wb') as f:
                pickle.dump({tok: weight for tok, weight in zip(placeholder_toks, weights)}, f)


        for weight, token_id in zip(weights, placeholder_token_ids):
            token_embeds[token_id] = torch.tensor(weight)


    elif args.adversarial:
        with open(args.adversarial_runs_file, 'r') as f:
            tmp = json.load(f)

        i = 0
        placeholder_toks = []
        adversarial_paths = {}
        for path in tmp:
            for epoch in tmp[path]:
                full_path = os.path.join(args.tokens_root, path, f'learned_embeds-{epoch}.bin')
                embedding_dict = torch.load(full_path, map_location='cpu')

                for value in embedding_dict.values():
                    placeholder_token = f'<new_token_{i}>'
                    token_info = {'epoch': epoch, 'embedding': value, 'placeholder_token': placeholder_token}
                    try:
                        adversarial_paths[full_path].append(token_info)
                    except KeyError:
                        adversarial_paths[full_path] = [token_info]
                    placeholder_toks.append(placeholder_token)
                    i += 1
                    
        # print(f'len before: {len(pipe.tokenizer)}; toks: {[placeholder_toks]}')
        num_added_tokens = pipe.tokenizer.add_tokens(placeholder_toks)
        # print(f'len after: {len(pipe.tokenizer)}')
        # print(f'npt: {len(placeholder_toks)} added: {num_added_tokens}')
        if num_added_tokens != len(placeholder_toks):
            raise ValueError('Not all the tokens were added!')

        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        token_embeds = pipe.text_encoder.get_input_embeddings().weight.data

        for full_path, value in adversarial_paths.items():
            for i, token_info in enumerate(value):
                placeholder_token = token_info['placeholder_token']
                placeholder_token_id = pipe.tokenizer.convert_tokens_to_ids(placeholder_token)
                adversarial_paths[full_path][i]['placeholder_token_id'] = placeholder_token_id
                token_embeds[placeholder_token_id] = torch.tensor(token_info['embedding'])

        # dump tokens for reproducibility
        os.makedirs(args.save_root, exist_ok=True)
        with open(os.path.join(args.save_root, f'tokens_{run_start_time}_{os.getpid()}.pkl'), 'wb') as f:
                pickle.dump(adversarial_paths, f)


    if args.adversarial or args.sampling is not None: # Log which images were generated by which run
        logs = {}
    set_seed(seed)
    # Dataset and DataLoaders creation:
    dataset = ImageDataset(data, args)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16)

    # Prepare everything with our `accelerator`.
    pipe, dataloader = accelerator.prepare(pipe, dataloader)
        

    total_batch_size = args.batch_size * accelerator.num_processes  
    logger.info("***** Running image generation *****")
    logger.info(f"  Num examples = {len(dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel) = {total_batch_size}")

    os.makedirs(args.save_root, exist_ok=True)
    with open(os.path.join(args.save_root, f'args_{run_start_time}_{os.getpid()}.json'), 'w') as f:
        json.dump({k: v for k, v in vars(args).items()}, f)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process, total=len(dataloader))
    progress_bar.set_description("Steps")

    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
    start_prev = timer()

    for step, batch in enumerate(dataloader):
        # TODO: that shouldn't work, depth_pil_inv_scaled and depth_path could be different size, but I call zip() later..
        depth_pil_inv_scaled = [Image.fromarray(x.cpu().numpy()) for x in batch["depth"]]

        depth_path = batch["depth_path"]
        rgb_batch = [Image.open(x.replace('depth_zbuffer','rgb')) for x in depth_path]

        current_batch_len = len(depth_pil_inv_scaled)
        negative_prompt = ["comic,cartoon,synthetic,rendered,animated,painting,sketch,drawing,highly saturated,person,people,face,body parts,eyes"] * current_batch_len

        if args.sampling is not None:
            token = np.random.choice(placeholder_toks)
            prompt = [f'{token},photo,highly detailed,photorealistic'] * current_batch_len
        elif args.adversarial:
            run = np.random.choice(list(adversarial_paths.keys()))
            token = "".join([value["placeholder_token"] for value in adversarial_paths[run]])
            prompt = [f'{token},photo,highly detailed,photorealistic'] * current_batch_len
        else:
            prompt = [args.prompt] * current_batch_len

        diff_start = timer()

        with torch.no_grad():
            
            if args.sdedit:
                images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=args.num_inference_steps, 
                            generator=generator, control_image=depth_pil_inv_scaled, image=rgb_batch, strength=args.strength,
                            controlnet_conditioning_scale=args.controlnet_conditioning_scale).images
            else:
                images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=args.num_inference_steps, 
                            generator=generator, image=depth_pil_inv_scaled,
                            controlnet_conditioning_scale=args.controlnet_conditioning_scale).images
            
        diff_end = timer()

        saved_imgs_paths = []
        for img, path in zip(images, depth_path):
            if not img.getbbox(): # not saving black images
                    continue
            img_save_dir = '/'.join(path.split('/')[-3:-1])
            img_save_file = path.split('/')[-1]
            img_save_path = f'{args.save_root}/{img_save_dir}/{img_save_file}'
            os.makedirs(f'{args.save_root}/{img_save_dir}'.replace('depth_zbuffer','rgb'), exist_ok=True)
            # print(f'saving image to {img_save_path}')
            img.save(img_save_path.replace('depth_zbuffer','rgb'))
            saved_imgs_paths.append(img_save_path.replace('depth_zbuffer','rgb'))

        if args.adversarial or args.sampling is not None:
            # ---- log saved imgs which paths ----
            try:
                logs[token].extend([path for path in saved_imgs_paths])
            except KeyError:
                logs[token] = [path for path in saved_imgs_paths]

        save_end = timer()
        progress_bar.set_description(f'Step {step}. Batch: {diff_start - start_prev:.3f}, Exec: {diff_end - diff_start:.3f}, Save: {save_end - diff_end:.3f} (sec); nis: {args.num_inference_steps}')
        progress_bar.update(1)
        start_prev = timer()

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()

    if args.adversarial or args.sampling is not None:
        with open(f'{args.save_root}/images-info_{run_start_time}_{os.getpid()}.json', 'w') as f:
            json.dump(logs, f)

if __name__ == "__main__":
    main()
