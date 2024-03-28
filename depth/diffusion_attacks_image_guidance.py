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
import logging
import math
import os
import random
import json
import pickle
import warnings
from pathlib import Path
from typing import Optional
from itertools import product

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami

from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

import diffusers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

import torchvision.transforms as tf
import torchvision.transforms.functional as tvf
import torchvision.utils as tvu

from utils import randn_tensor, get_depth_pil, get_batch_from_pils, resize_img, prepare_images_for_clip
from models.unet import UNet_taskonomy
from midas.dpt_depth import DPTDepthModel
from losses import midas_loss
from datasets import datasets


if is_wandb_available():
    import wandb

# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__)

midas_loss_ = midas_loss.MidasLoss(alpha=0.1)

def log_validation(text_encoder, tokenizer, controlnet, unet, vae, target_model, prompt, args, accelerator, weight_dtype, train_points=None, size=256):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {prompt}."
    )

    if accelerator.is_main_process: # ty: dont know why i need to add this, it doesnt work otherwise
        if args.dataset == 'taskonomy':
            conditioning_data_dir = os.path.join(args.taskonomy_data_root, f'{args.building}_depth_zbuffer/depth_zbuffer/')
            conditioning_data_dir_val = os.path.join(args.taskonomy_data_root, f'{args.building}_depth_zbuffer/depth_zbuffer/')
        else:
            conditioning_data_dir = os.path.join(args.taskonomy_data_root, f'depth_zbuffer/{args.dataset}/{args.building}')
            conditioning_data_dir_val = os.path.join(args.taskonomy_data_root, f'depth_zbuffer/{args.dataset}/{args.building_val}')
            
        pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            controlnet=controlnet,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )   
        pipeline.scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        pipeline = pipeline.to(accelerator.device)
        pipeline.safety_checker = None
        pipeline.set_progress_bar_config(disable=True)
        
        # run inference
        generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
        negative_prompt = args.negative_prompt if args.guidance_scale > 1. else ""
        images_val = []

        if args.dataset in ['taskonomy','replica']:
            points_views = product([10, 11], [0,2])
        elif args.dataset in ['hypersim']:
            points_views = [[0,0],[10,0],[83,0],[67,0]]
        elif args.dataset in ['replica_gso']:
            points_views = [[101,2],[1050,0],[400,1],[997,8]]
        elif args.dataset in ['blended_mvg']:
            points_views = [[0,0],[38,0],[8,0],[46,0]]

        for (point_id, view_id) in points_views:
            if args.dataset == 'taskonomy':
                depth_path = os.path.join(args.taskonomy_data_root, f'albertville_depth_zbuffer/depth_zbuffer/point_{point_id}_view_{view_id}_domain_depth_zbuffer.png')
            else:
                depth_path = os.path.join(conditioning_data_dir_val, f'point_{point_id}_view_{view_id}_domain_depth_zbuffer.png')
            _, depth_pil_inv_scaled = get_depth_pil(depth_path, args.dataset)
            
            inputs = depth_pil_inv_scaled
            rgb_img = Image.open(depth_path.replace('depth_zbuffer','rgb'))
            images_val.append(rgb_img)
            for _ in range(args.num_validation_images):
                with torch.autocast("cuda"):
                    image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=15, 
                                     generator=generator, control_image=inputs, image=rgb_img, strength=args.strength).images[0]
                images_val.append(image.resize((size,size)))

        rgb_batch = get_batch_from_pils(images_val,resize=size)

        with torch.no_grad():
            preds_val = target_model(rgb_batch.to(accelerator.device).to(dtype=weight_dtype)).cpu()
        if preds_val.dim() == 3: preds_val = preds_val.unsqueeze(1)

        if args.target_model == 'xtc_depth':
            preds_val[:,0,0,0] = 0. # ty: this is a hack to get wandb to not scale the grayscale image to 0-1
            preds_val[:,0,0,1] = 1.

        # TODO: hardcoded dimensions..
        preds_val = preds_val.view(args.num_validation_images + 1, 4 , 1, size, size)
        preds_val = preds_val.view(-1, 1, size, size)
        preds_val_grid = tvu.make_grid(preds_val.float(),nrow=args.num_validation_images+1)
        rgb_val_grid = tvu.make_grid(rgb_batch,nrow=args.num_validation_images+1)

        if train_points is None:
            train_points = [{'point_id': 4, 'view_id': 1}, {'point_id': 4, 'view_id': 2},
                            {'point_id': 5, 'view_id': 1}, {'point_id': 5, 'view_id': 2}]

        images_train = []
        gt_train = []
        for point in train_points:
            point_id = point['point_id']
            view_id = point['view_id']
            depth_path = os.path.join(conditioning_data_dir, f'point_{point_id}_view_{view_id}_domain_depth_zbuffer.png')
            _, depth_pil_inv_scaled = get_depth_pil(depth_path, args.dataset)
            inputs = depth_pil_inv_scaled
            rgb_img = Image.open(depth_path.replace('depth_zbuffer','rgb'))
            images_train.append(rgb_img)

            for _ in range(args.num_validation_images):
                with torch.autocast("cuda"):
                    image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=args.num_inference_steps, generator=generator, 
                                     control_image=inputs, image=rgb_img,  strength=args.strength).images[0]
                images_train.append(image.resize((size,size)))

        rgb_batch = get_batch_from_pils(images_train,resize=size)
        
        with torch.no_grad():
            preds_train = target_model(rgb_batch.to(accelerator.device).to(dtype=weight_dtype)).cpu()
        
        preds_train = preds_train.view(args.num_validation_images + 1, len(train_points) , 1, size, size)
        preds_train = preds_train.view(-1, 1, size, size)

        if preds_train.dim() == 3: preds_train = preds_train.unsqueeze(1)
        if args.target_model == 'xtc_depth':
            preds_train[:,0,0,0] = 0.
            preds_train[:,0,0,1] = 1.
        preds_train_trainparams_grid = tvu.make_grid(preds_train.float(),nrow=args.num_validation_images+1)
        rgb_train_trainparams_grid = tvu.make_grid(rgb_batch,nrow=args.num_validation_images+1)

        images_train = []
        for point in train_points:
            point_id = point['point_id']
            view_id = point['view_id']
            depth_path = os.path.join(conditioning_data_dir, f'point_{point_id}_view_{view_id}_domain_depth_zbuffer.png')
            _, depth_pil_inv_scaled = get_depth_pil(depth_path, args.dataset)
            inputs = depth_pil_inv_scaled
            rgb_img = Image.open(depth_path.replace('depth_zbuffer','rgb'))
            images_train.append(rgb_img)
            for _ in range(args.num_validation_images):
                with torch.autocast("cuda"):
                    image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=15, generator=generator, 
                                     control_image=inputs, image=rgb_img, strength=args.strength).images[0]
                images_train.append(image.resize((size,size)))

        rgb_batch = get_batch_from_pils(images_train,resize=size)
        with torch.no_grad():
            preds_train = target_model(rgb_batch.to(accelerator.device).to(dtype=weight_dtype)).cpu()
        if preds_train.dim() == 3: preds_train = preds_train.unsqueeze(1)

        if args.target_model == 'xtc_depth':
            preds_train[:,0,0,0] = 0.
            preds_train[:,0,0,1] = 1.
        preds_train = preds_train.view(args.num_validation_images + 1, len(train_points) , 1, size, size)
        preds_train = preds_train.view(-1, 1, size, size)
        preds_train_grid = tvu.make_grid(preds_train.float(),nrow=args.num_validation_images+1)
        rgb_train_grid = tvu.make_grid(rgb_batch,nrow=args.num_validation_images+1)

        img_preds_trainparams = preds_train_trainparams_grid
        img_preds_train = preds_train_grid
        img_preds_val = preds_val_grid
        
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                logs = {
                        "train-trainparams": [wandb.Image(resize_img(rgb_train_trainparams_grid), caption=prompt)],
                        "predictions-train-trainparams": [wandb.Image(resize_img(img_preds_trainparams))],
                        "train": [wandb.Image(resize_img(rgb_train_grid), caption=prompt)],
                        "predictions-train": [wandb.Image(resize_img(img_preds_train))],
                        "validation": [wandb.Image(resize_img(rgb_val_grid), caption=prompt)],
                        "predictions-validation": [wandb.Image(resize_img(img_preds_val))]
                    }
                tracker.log(logs)

        del pipeline
    torch.cuda.empty_cache()


def save_progress(text_encoder, accelerator, args, save_path, learned_embeds=None):
    logger.info("Saving embeddings")
    learned_embeds_dict = {}
    if learned_embeds is None:
        learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[49408:]
    for i in range(args.num_new_tokens):
        learned_embeds_dict[f'<placeholder_token_{i}>'] = learned_embeds[i].detach().cpu()
    torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    
    parser.add_argument(
        "--taskonomy_data_root", type=str, default=None, required=True, help="A folder containing the data root."
    )

    parser.add_argument(
        "--taskonomy_split_path", type=str, default=None, help="A folder containing the trainval split (for sample different buildings)."
    )
    parser.add_argument(
        "--mask_data_root", type=str, default=None, required=True, help="A folder containing the masks."
    )

    parser.add_argument(
        "--building", type=str, default=None, required=True, help="Name of taskonomy building used for training."
    )

    parser.add_argument(
        "--building_val", type=str, default=None, required=True, help="Name of building used for val."
    )

    parser.add_argument(
        "--dataset", 
        type=str,
        default='taskonomy',
        choices=['taskonomy', 'replica', 'replica_gso', 'hypersim', 'blended_mvg'],
        help="dataset to optimize over"
    )

    parser.add_argument(
        "--data_paths_file", 
        type=str,
        default=None,
        nargs='+',
        help="File with the data paths."
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        help="whether to run in debug mode (used for wandb grouping).",
    )

    parser.add_argument(
        "--strength", type=float, default=1.0, help="How much noise to apply to initial image."
    )

    parser.add_argument(
        "--image_point",
        type=str,
        default=None,
        help=(
            "For `granularity=image`, the point we want to select an image from. If not passed, selected randomly form the dataset."
        ),
    )

    parser.add_argument(
        "--image_view",
        type=str,
        default=None,
        help=(
            "For `granularity=image`, the view we want to select an image from. If not passed, selected randomly form the dataset."
        ),
    )

    parser.add_argument(
        "--initializer_token", type=str, default=None, help="A token to use as initializer word."
    )
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.0, help="Guidance scale."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
        help=(
            "Number of inference steps to run during training."
        ),
    )

    parser.add_argument(
        "--num_new_tokens",
        type=int,
        default=None,
        required=True,
        help="Number of new tokens to add (if use hard prompts)",
    )

    parser.add_argument(
        "--initialize_emb_random",
        action='store_true',
        help="Initialize tokens randomly (default value: room)",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    
    parser.add_argument(
        "--target_model",
        type=str,
        default='xtc_depth',
        help="Target model.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--rgb2depth_checkpoint_path",
        type=str,
        default=None,
        help="Path to the rgb2depth checkpoint.",
    )

    parser.add_argument(
        "--pretrained_tokens_path",
        type=str,
        default=None,
        help="Path to pretrained adversarial tokens.",
    )

    parser.add_argument(
        "--pretrained_tokens_sample_file",
        type=str,
        default=None,
        help="Path to a json containing previous adv runs. Used to sample token embeddings for the run.",
    )


    parser.add_argument(
        "--pretrained_stats_file",
        type=str,
        default=None,
        help="Path to a pickle containing previous adv runs stats. Used to sample token embeddings for the run.",
    )

    parser.add_argument(
        "--pretrained_tokens_sample_mode",
        type=str,
        default=None,
        choices=["mean_emb_cov_adv", "mean_adv_cov_adv", 'mean_emb_cov_emb', 'no'],
        help="Init sampling mode",
    )

    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="comic,cartoon,synthetic,rendered,animated,painting,sketch,drawing,highly saturated,person,people,face,body parts,eyes",
        help="A negative prompt that is used during validation to verify that the model is learning.",
    )

    parser.add_argument(
        "--add_default_prompt",
        action='store_true',
        help="to add the default prompts photorealistic etc to all prompts.",
    )

    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default=None,
        choices=['l1', 'midas'],
        help=("Loss function used for training."),
    )

    parser.add_argument("--early_stop", action='store_true')
    parser.add_argument("--early_stop_threshold", type=float, help='early stopping loss threshold')
    parser.add_argument("--running_avg_size", type=int, default=1, help='size of the window for calculating running average')
    parser.add_argument("--iter", type=int, default=1, help='Iteration for logging.')

    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )

    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=100,
        help=(
            "Run only N batches of the validation dataset."
        ),
    )

    parser.add_argument('--clip_text_guidance', default="", type=str)
    parser.add_argument('--clip_text_guidance_coef', default=1., type=float)

    parser.add_argument('--remove_adv_loss', action='store_true', help='do not do adversarial optimization.')

    parser.add_argument('--clip_image_guidance', action='store_true')
    parser.add_argument('--clip_image_guidance_coef', default=1., type=float)
    parser.add_argument('--target_domain_images_path_file', type=str, default=None)
    parser.add_argument('--turn_on_percep_steps', type=int, default=-1)
    

    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )
    # breakpoint()
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process and args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # breakpoint()
    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load scheduler and models
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")

    # # load depth model
    if args.rgb2depth_checkpoint_path is None:
        model_path = '/datasets/home/consistency_final_models/rgb2depth_consistency.pth'
    else:
        model_path = args.rgb2depth_checkpoint_path
    sd = torch.load(model_path)
    if 'omnidata' in args.rgb2depth_checkpoint_path:
        sd = sd["state_dict"]
        rgb2depth = DPTDepthModel().to(accelerator.device)
    else:
        rgb2depth = UNet_taskonomy(downsample=6, out_channels=1).to(accelerator.device)

    # TODO: fix that...
    try:
        rgb2depth.load_state_dict(sd)
    except RuntimeError:
        if 'omnidata' in args.rgb2depth_checkpoint_path:
            new_dict = {k[6:]:v for k,v in sd.items()}
        else:
            new_dict = {}
            for k, v in sd["('rgb', 'depth_zbuffer')"].items():
                new_k = '.'.join(k.split('.')[3:])
                new_dict[new_k] = v
        rgb2depth.load_state_dict(new_dict)
    rgb2depth.eval()

    # ---- load supermodel ----
    supermodel_dir = '/datasets/home/oguzhan/ainaz_supermodels/final_demo_models_190422/depth/'
    sd = torch.load(f'{supermodel_dir}/graph.pth')
    sd = sd["('rgb', 'normal')"]
    sd = {k[28:]:v for k,v in sd.items()}
    supermodel = DPTDepthModel() #.to(accelerator.device)
    supermodel.load_state_dict(sd)
    supermodel.eval()

    conditioning_data_dir = os.path.join(args.taskonomy_data_root, f'{args.building}_depth_zbuffer/depth_zbuffer')
    mask_data_root = os.path.join(args.mask_data_root, f'{args.building}_mask_valid')
    
    # Dataset and DataLoaders creation:
    dataset = datasets.get_dataset(
        args.dataset,
        cond_data_root=conditioning_data_dir,
        mask_data_root=mask_data_root,
        tokenizer=tokenizer,
        size=args.resolution,
        pred_size=384 if args.target_model in ['supermodel_depth','dpt_depth'] else 256,
        repeats=args.repeats,
        center_crop=args.center_crop,
        negative_prompt=args.negative_prompt,
        num_new_tokens=args.num_new_tokens,
        data_paths_file=args.data_paths_file,
        add_default_prompt=args.add_default_prompt,
        taskonomy_split_path=args.taskonomy_split_path,
    )

    train_dataloader = dataset.get_train_dataloader(batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    val_dataloader = dataset.get_validation_dataloader(batch_size=1, shuffle=True, num_workers=args.dataloader_num_workers)

    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataloader)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataloader)

    # Add the placeholder token in tokenizer
    new_tokens = [f'<placeholder_token_{i}>' for i in range(args.num_new_tokens)]
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    # todo: when I will add loading trained tokens, that might fail
    if num_added_tokens != args.num_new_tokens:
        raise ValueError('Tokens are added incorrectly.')
        
    new_token_ids = tokenizer.convert_tokens_to_ids(new_tokens)
    text_encoder.resize_token_embeddings(len(tokenizer))

    token_embeds = text_encoder.get_input_embeddings().weight.data

    start_emb_norm = token_embeds.cpu().norm(2, dim=1).mean()

    if args.initializer_token is None:
        raise ValueError('While not using hard prompts, you should pass initializer_token.')
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    init_token_embed = token_embeds[initializer_token_id].to(accelerator.device)
    
    if args.pretrained_tokens_sample_file: # --- initialize with a distribution over adv runs

        with open(args.pretrained_tokens_sample_file, 'r') as f:
            tmp = json.load(f)

        adv_embeddings = []
        for path in tmp:
            for epoch in tmp[path]:
                full_path = f'/scratch/alekseev/work/experiments/diff_dataset/{path}/learned_embeds-{epoch}.bin'
                embedding = torch.load(full_path)['<placeholder_token_0>'] # todo: may be not only "0"
                adv_embeddings.append(embedding)

        accelerator.print(f'sampling from {args.pretrained_tokens_sample_mode}')
        if args.pretrained_tokens_sample_mode == 'mean_emb_cov_adv':
            mean = token_embeds.cpu().numpy().mean(axis=0)
            cov = np.cov(torch.stack(adv_embeddings).numpy(), rowvar=0)
        elif args.pretrained_tokens_sample_mode == 'mean_adv_cov_adv':
            mean = np.mean(torch.stack(adv_embeddings).numpy(), axis=0)
            cov = np.cov(torch.stack(adv_embeddings).numpy(), rowvar=0)

        weights = np.random.multivariate_normal(mean=mean, cov=cov, size=args.num_new_tokens)
        for new_token_id, weight in zip(new_token_ids, weights):
            token_embeds[new_token_id] = torch.tensor(weight)

    elif args.pretrained_tokens_sample_mode == 'mean_emb_cov_emb':
        accelerator.print(f'sampling from {args.pretrained_tokens_sample_mode}')
        mean = token_embeds.cpu().numpy().mean(axis=0)
        cov = np.cov(token_embeds.cpu().numpy(), rowvar=0)
        weights = np.random.multivariate_normal(mean=mean, cov=cov, size=args.num_new_tokens)
        for new_token_id, weight in zip(new_token_ids, weights):
            token_embeds[new_token_id] = torch.tensor(weight)

    elif args.pretrained_stats_file != 'no' or args.pretrained_stats_file is not None:
        with open(args.pretrained_stats_file, 'rb') as f:
            stats = pickle.load(f)

        mean = stats['mean']
        cov = stats['cov']

        accelerator.print('sampling from pretrained stats file')
        weights = np.random.multivariate_normal(mean=mean, cov=cov, size=args.num_new_tokens)
        for new_token_id, weight in zip(new_token_ids, weights):
            token_embeds[new_token_id] = torch.tensor(weight)
            
    elif args.pretrained_tokens_path: # --- initialize with a prev adv run
            learned_embeds_dict = torch.load(args.pretrained_tokens_path)
            if list(learned_embeds_dict.keys()) != [f'<placeholder_token_{i}>' for i in range(args.num_new_tokens)]:
                raise ValueError(f'Pretrained tokens ({len(learned_embeds_dict)}) do not correspond to current tokens ({args.num_new_tokens})')
            for new_token, new_token_id in zip(new_tokens, new_token_ids):
                token_embeds[new_token_id] = learned_embeds_dict[new_token]
    else: # --- initialize with some emb tokens
        if args.initialize_emb_random: # --- random
            init_token_ids = random.sample(list(range(0, 49408)), k=args.num_new_tokens)
            accelerator.print(f'initialized tokens: {[tokenizer.decode(tid) for tid in init_token_ids]}')

            for new_token_id, init_token_id in zip(new_token_ids, init_token_ids):
                token_embeds[new_token_id] = token_embeds[init_token_id]
        else: # --- room
            for new_token_id in new_token_ids:
                token_embeds[new_token_id] = token_embeds[initializer_token_id]


    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)
    rgb2depth.requires_grad_(False)
    supermodel.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    rgb2depth.to(accelerator.device, dtype=weight_dtype)
    supermodel.to(accelerator.device, dtype=weight_dtype)

    if args.clip_image_guidance:

        if args.target_domain_images_path_file is None:
            raise ValueError(f"Need to precise target domain image paths to use this perceptual loss {args.perceptual_loss}")
        
        ## load clip model
        perceptual_loss_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        perceptual_loss_model.to(accelerator.device, dtype=weight_dtype)
        perceptual_loss_model.requires_grad_(False)
        
        with open(args.target_domain_images_path_file, 'r') as f:
            image_paths = [x.strip() for x in f.readlines()]

        images = [PIL.Image.open(image_path) for image_path  in image_paths]
                            ## list[pil] -> B C H W

        pixel_values = prepare_images_for_clip(images=images, resolution=224)
        pixel_values = pixel_values.to(accelerator.device, dtype=weight_dtype)

        with torch.no_grad():
            features = perceptual_loss_model(pixel_values).pooler_output

        mean_target_clip_features = features.mean(dim=0)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("diffusion-attacks" if 'xtc' in args.target_model else "diffusion-attacks-depth-dpt", config=vars(args), init_kwargs={"wandb":{"name":args.output_dir.split('/')[-1], "entity": "diff-dataset"}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    do_classifier_free_guidance = args.guidance_scale > 1.0
    
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    extra_step_kwargs = {}
    extra_step_kwargs['generator'] = generator
    noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device)
    # timesteps = noise_scheduler.timesteps
    # num_warmup_steps = len(timesteps) - args.num_inference_steps * noise_scheduler.order

    def get_timesteps(num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = noise_scheduler.timesteps[t_start * noise_scheduler.order :]

        return timesteps, num_inference_steps - t_start
    
    timesteps, num_inference_steps = get_timesteps(args.num_inference_steps, args.strength, accelerator.device)
    latent_timestep = timesteps[:1].repeat(args.train_batch_size)
    print("denoising for this number of timesteps: ",timesteps)

    def prepare_latents(image, timestep, batch_size, dtype, device, generator=None, logger=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )
        
        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4: # if image is latent
            init_latents = image
        else:
            init_latents = vae.encode(image).latent_dist.sample(generator)
            init_latents = vae.config.scaling_factor * init_latents

        shape = init_latents.shape
        noise = randn_tensor(shape, logger, generator=generator, device=device, dtype=dtype)
        # if strength is 1. then initialise the latents to noise, else initial to image + noise
        init_latents = noise if args.strength==1 else noise_scheduler.add_noise(init_latents, noise, timestep)
        # if pure noise then scale the initial latents by the  Scheduler's init sigma
        init_latents = init_latents * noise_scheduler.init_noise_sigma if args.strength==1 else init_latents

        latents = init_latents

        return latents

    with torch.no_grad():
        val_loss = validate(val_dataloader,
            prepare_latents, rgb2depth, progress_bar,
            text_encoder, unet, noise_scheduler, controlnet, vae,
            args, accelerator, generator, weight_dtype, timesteps,
            global_step, latent_timestep, extra_step_kwargs)

    val_loss_changed = True

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    early_stop = False
    running_loss = []
    prev_embeds = []
    if args.clip_text_guidance:
        path = args.clip_text_guidance.split(':', 1)[1]
        clip_text_guidance_emb = torch.load(path, map_location=accelerator.device).to(dtype=weight_dtype)
    
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            
            with accelerator.accumulate(text_encoder):
                logs = {}
                loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                prompt_embeds = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                # breakpoint()
                if args.clip_text_guidance:                        
                    if args.clip_text_guidance.startswith('l2'):
                        text_guidance_loss = (prompt_embeds - clip_text_guidance_emb).pow(2).mean()
                    elif args.clip_text_guidance.startswith('cosine'):
                        text_guidance_loss = 1 - F.cosine_similarity(prompt_embeds, clip_text_guidance_emb).mean()

                    logs['text_guidance_loss'] = text_guidance_loss.detach().item()

                    loss = args.clip_text_guidance_coef * text_guidance_loss

                negative_prompt_embeds = text_encoder(batch["neg_input_ids"])[0].to(dtype=weight_dtype)
                if do_classifier_free_guidance:
                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                    batch["conditioning"] = batch["conditioning"].repeat(2,1,1,1)

                # 6. Prepare latent variables
                # rgb image should be normalized to [-1,1] based on this:
                # https://github.com/huggingface/diffusers/blob/29f15673ed5c14e4843d7c837890910207f72129/src/diffusers/pipelines/controlnet/pipeline_controlnet_img2img.py#L198
                latents = prepare_latents(
                    batch["rgb"],
                    latent_timestep,
                    args.train_batch_size,
                    prompt_embeds.dtype,
                    accelerator.device,
                    generator,
                    logger,
                ).to(dtype=weight_dtype)
                # breakpoint()

                # ty: from https://github.com/huggingface/diffusers/blob/8b451eb63b0f101e7fcc72365fe0d683808b22cd/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_controlnet.py#L931-L994
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

                    # controlnet(s) inference
                    conditioning = batch["conditioning"].to(dtype=weight_dtype)
                    conditioning_scale = 1.
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=conditioning,
                        conditioning_scale=conditioning_scale,
                        return_dict=False,
                    )
                    
                    # predict the noise residual
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        # cross_attention_kwargs=cross_attention_kwargs,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample


                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # 8. Post-processing
                latents = 1 / vae.config.scaling_factor * latents
                image = vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)

                if args.target_model == 'xtc_depth': # ty: xtc model was trained with 256x256 images
                    image = tvf.resize(image,256,PIL.Image.BILINEAR)
                    gt = tvf.resize(batch['gt'],256,PIL.Image.BILINEAR)
                    cond = tvf.resize(batch["conditioning"],256,PIL.Image.BILINEAR)
                elif args.target_model in ['supermodel_depth','dpt_depth']: # ty: supermodel was trained with 384x384 images
                    image = (image - 0.5) / 0.5
                    gt = tvf.resize(batch['gt'],384,PIL.Image.BILINEAR)
                    cond = batch["conditioning"]


                if global_step % 20 == 0:
                    viz_img = image
                    if args.target_model in ['supermodel_depth','dpt_depth']: viz_img = 0.5 * image + 0.5
                    viz_orig_rgb = tvf.resize(batch["rgb"][:args.train_batch_size],viz_img.size(-1),PIL.Image.BILINEAR)
                    img_depth = torch.cat([viz_img,cond[:args.train_batch_size],(viz_orig_rgb + 1) * 127.5/255.],dim=0)
                    log_img_depth = tvu.make_grid(img_depth.cpu().detach().float(), nrow=args.train_batch_size, padding=5, pad_value=1)
                    log_img_depth = wandb.Image(resize_img(log_img_depth), caption="train_generation")
                
                preds = rgb2depth(image)
                with torch.no_grad():
                    preds_supermodel = supermodel(image)
                
                # ---- loss ----
                if args.loss == 'l1':
                    adv_loss = -((preds-gt) * batch['mask']).abs().mean()  # l1 loss
                elif args.loss == 'midas':
                    _, loss_midas = midas_loss_(preds.float(), gt[:,:1], batch['mask'].bool())  # scale shift invariant loss
                    adv_loss = -1. * loss_midas
                else:
                    raise ValueError(f'loss {args.loss} is not supported.')

                if args.remove_adv_loss:
                    adv_loss = torch.tensor(1.)
                else:
                    loss += adv_loss
                    logs.update({"adv_loss": adv_loss.detach().item()})
                
                running_loss.append(adv_loss.detach().cpu().numpy().item())
                if len(running_loss) > args.running_avg_size * args.gradient_accumulation_steps:
                    running_loss.pop(0)
                
                ## compute perceptual loss!
                if args.clip_image_guidance:
                    clip_processed_image = prepare_images_for_clip(images=image, resolution=224).to(accelerator.device, dtype=weight_dtype)
                    clip_features = perceptual_loss_model(clip_processed_image).pooler_output
                    perceptual_loss = 1 - F.cosine_similarity(mean_target_clip_features, clip_features).mean()
                    
                    coef = args.clip_image_guidance_coef if global_step > args.turn_on_percep_steps else 0

                    loss +=  coef * perceptual_loss

                    logs.update({"visual_guidance_loss": perceptual_loss.detach().item()})
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.unscale_gradients(optimizer=optimizer)
                    grad_norm = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.grad[new_token_ids].detach().cpu().norm(2)
                # breakpoint()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # ---- all losses ----
                gt_supermodel = tvf.resize(batch['gt'],preds_supermodel.size(-1),PIL.Image.BILINEAR)
                mask_supermodel = tvf.resize(batch['mask'],preds_supermodel.size(-1),PIL.Image.BILINEAR)
                with torch.no_grad():
                    loss_l1 = ((preds-gt) * batch['mask']).abs().mean()  # l1 loss
                    supermodel_loss_l1 = ((preds_supermodel - gt_supermodel) * mask_supermodel).abs().mean()  # l1 loss
                    _, loss_midas = midas_loss_(preds.float(), gt[:,:1], batch['mask'].bool())  # scale shift invariant loss
                    _, supermodel_loss_midas = midas_loss_(preds_supermodel.float(), gt_supermodel[:,:1], mask_supermodel.bool())  # scale shift invariant loss

                logs["loss"] = loss.detach().item()
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                logs["supermodel_loss_l1"] = supermodel_loss_l1.cpu().detach().item()
                logs[ "supermodel_loss_midas"] = supermodel_loss_midas.cpu().detach().item()
                logs["loss_l1"] = loss_l1.cpu().detach().item()
                logs["loss_midas"] = loss_midas.cpu().detach().item()

                if accelerator.sync_gradients:
                    logs.update({"grad_norm": grad_norm.item()})
                
                # Let's make sure we don't update any embedding weights besides the newly added token
                with torch.no_grad():
                    index_no_updates = torch.arange(0, 49408) 
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

                    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data[49408:]
                    learned_embeds = learned_embeds.detach().cpu()

                    logs.update({"learned_emb_mean": learned_embeds.mean().item(),
                    "learned_emb_std": learned_embeds.std(dim=1).mean().item(),
                    "learned_emb_L2norm": learned_embeds.norm(2, dim=1).mean().item()})
               
                if accelerator.is_main_process:
                    orig_rgb =  tvf.resize(batch["rgb"][:args.train_batch_size],viz_img.size(-1),PIL.Image.BILINEAR)
                    with torch.no_grad(): 
                        if args.target_model == 'xtc_depth':
                            orig_rgb = (orig_rgb + 1) * 127.5/255. # use viz_orig_rgb because it has been resized
                        preds = rgb2depth(orig_rgb.to(dtype=weight_dtype))  
                    if args.loss == 'l1':
                        loss_orig = ((preds-gt) * batch['mask']).abs().mean()  # l1 loss
                    elif args.loss == 'midas':
                        _, loss_orig = midas_loss_(preds.float(), gt[:,:1], batch['mask'].bool())
                    
                    logs.update({"loss_orig": loss_orig.item()})
                if global_step % 20 == 0:
                    logs.update({"train_generation": log_img_depth})

                if val_loss_changed:
                    logs.update({"mean_validation_loss": np.mean(val_loss)})
                    val_loss_changed = False
                progress_bar.set_postfix(**logs)
                progress_bar.set_description(f'Training. Epoch {epoch}')

                accelerator.log(logs, step=global_step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # ---- Early stop logic ----

                prev_embeds.append(learned_embeds)
                if len(prev_embeds) > args.running_avg_size:
                    prev_embeds.pop(0)
                    
                if args.early_stop:
                    # print(f'len: {len(running_loss)}, loss: {np.mean(running_loss)}; T: {args.early_stop_threshold}')
                    
                    if len(running_loss) == args.running_avg_size * args.gradient_accumulation_steps and np.mean(running_loss) < args.early_stop_threshold:
                        accelerator.print('Early stopping!')

                        # set prev embeds
                        with torch.no_grad():
                            accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[49408:] = prev_embeds[0]

                            validation_prompt = ','.join([f'<placeholder_token_{i}>' for i in range(args.num_new_tokens)]) 
                            if args.add_default_prompt:
                                validation_prompt += ',photo,highly detailed,photorealistic'
                                
                            log_validation(text_encoder, tokenizer, controlnet, unet, vae, rgb2depth, validation_prompt, args, 
                                           accelerator, weight_dtype, [{'point_id': args.image_point, 'view_id': args.image_view}], size=384 if args.target_model in ['supermodel_depth','dpt_depth'] else 256,)

                        save_path = os.path.join(args.output_dir, f"learned_embeds-{global_step}.bin")
                        save_progress(text_encoder, accelerator, args, save_path, learned_embeds=prev_embeds[0])
                        try:
                            a = wandb.Artifact(f"learned_embeds-{global_step}", type='tensor', metadata={'step': global_step})
                            a.add_file(local_path=save_path)
                            wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
                            wandb_tracker.log_artifact(a)
                        except AttributeError:
                            accelerator.print('failed to save artifact')

                        early_stop = True
                # ---- Early stop logic end ----

                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"learned_embeds-{global_step}.bin")
                    save_progress(text_encoder, accelerator, args, save_path)
                    
                    if np.mean(running_loss) > 0.07:
                        save_progress(text_encoder, accelerator, args, os.path.join(args.output_dir, f"cool-learned_embeds-{global_step}.bin"))

                    save_progress(text_encoder, accelerator, args, save_path)
                    try:
                        a = wandb.Artifact(f"learned_embeds-{global_step}", type='tensor', metadata={'step': global_step})
                        a.add_file(local_path=save_path)
                        wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
                        wandb_tracker.log_artifact(a)
                    except AttributeError:
                        accelerator.print('failed to save artifact')

                
                if global_step % args.validation_steps == 0:
                    # Validation is done on all gpus
                    with torch.no_grad():
                        val_loss = validate(val_dataloader,
                            prepare_latents, rgb2depth, progress_bar,
                            text_encoder, unet, noise_scheduler, controlnet, vae,
                            args, accelerator, generator, weight_dtype, timesteps,
                            global_step, latent_timestep, extra_step_kwargs)
                    
                    val_loss_changed = True
                    if accelerator.is_main_process:  # Image generation only on one gpu (main)
                        validation_prompt = ','.join([f'<placeholder_token_{i}>' for i in range(args.num_new_tokens)]) 
                        if args.add_default_prompt:
                            validation_prompt += ',photo,highly detailed,photorealistic'
                        log_validation(text_encoder, tokenizer, controlnet, unet, vae, rgb2depth, validation_prompt, args, 
                                       accelerator, weight_dtype, [{'point_id': args.image_point, 'view_id': args.image_view}], size=384 if args.target_model in ['supermodel_depth','dpt_depth'] else 256)

            if global_step >= args.max_train_steps or early_stop:
                break
    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            safety_checker = None,
            requires_safety_checker = False
        )
        pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        save_path = os.path.join(args.output_dir, "learned_embeds.bin")
        save_progress(text_encoder, accelerator, args, save_path)

    accelerator.end_training()


def validate(val_dataloader, 
             prepare_latents, rgb2depth, progress_bar,
             text_encoder, unet, noise_scheduler, controlnet, vae,
             args, accelerator, generator, weight_dtype, timesteps,
             global_step, latent_timestep,
             extra_step_kwargs,):
    
    losses = []

    do_classifier_free_guidance = args.guidance_scale > 1.0

    logger.info(f"Running validation on the val_dataloader for {args.max_val_batches} batches.")
    for step, batch in enumerate(val_dataloader):
        if step >= args.max_val_batches:
            break
    
        # This is not needed here since we're not calculating grads (?)
        prompt_embeds = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
        negative_prompt_embeds = text_encoder(batch["neg_input_ids"])[0].to(dtype=weight_dtype)
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            batch["conditioning"] = batch["conditioning"].repeat(2,1,1,1)

        # 6. Prepare latent variables
        num_channels_latents = unet.config.in_channels
        latents = prepare_latents(
            batch["rgb"],
            latent_timestep,
            args.train_batch_size,
            prompt_embeds.dtype,
            accelerator.device,
            generator,
            logger,
        ).to(dtype=weight_dtype)
        if batch["rgb"].size(0):   # upgrading diffusers makes add_noise step return noisy sample with same bs as training, not as input bs
            latents = latents[:1]

        # ty: from https://github.com/huggingface/diffusers/blob/8b451eb63b0f101e7fcc72365fe0d683808b22cd/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_controlnet.py#L931-L994
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            # breakpoint()
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

            # controlnet(s) inference
            conditioning = batch["conditioning"].to(dtype=weight_dtype)
            conditioning_scale = 1.
            
            down_block_res_samples, mid_block_res_sample = controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=conditioning,
                conditioning_scale=conditioning_scale,
                return_dict=False,
            )
            
            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                # cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # 8. Post-processing
        latents = 1 / vae.config.scaling_factor * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        if args.target_model == 'xtc_depth': # ty: xtc model was trained with 256x256 images
            image = tvf.resize(image,256,PIL.Image.BILINEAR)
            gt = tvf.resize(batch['gt'],256,PIL.Image.BILINEAR)
        elif args.target_model in ['supermodel_depth','dpt_depth']: # ty: supermodel was trained with 384x384 images
            gt = tvf.resize(batch['gt'],384,PIL.Image.BILINEAR)
        
        preds = rgb2depth(image)
        
        if args.loss == 'l1':
            loss = -((preds-gt) * batch['mask']).abs().mean()  # l1 loss
        elif args.loss == 'midas':
            _, loss = midas_loss_(preds.float(), gt[:,:1], batch['mask'].bool())  # scale shift invariant loss
            loss = -1. * loss
        else:
            raise ValueError(f'loss {args.loss} is not supported.')
        losses.append(loss.detach().item())
        

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # image = self.decode_latents(latents)
        # 9. Run safety checker
        # image, has_nsfw_concept = self.run_safety_checker(image, accelerator.device, prompt_embeds.dtype)

        progress_bar.set_description(f'Validation. Steps: {step} / {args.max_val_batches}')
        
        # logs = {"validation_loss": loss.detach().item()}
        # progress_bar.set_postfix(**logs)
        # accelerator.log(logs, step = global_step * args.max_val_batches + step)

    # logs = {"mean_validation_loss": np.mean(losses)}
    # progress_bar.set_postfix(**logs)
    # accelerator.log(logs, step = global_step + 10)
    torch.cuda.empty_cache()
    return losses


def infer_initial_losses(dataloader, rgb2depth, accelerator, progress_bar, args, weight_dtype, data_type='train'):
    
    losses = []

    logger.info(f"Running initial loss calculation for {data_type} dataloader.")
    for step, batch in enumerate(dataloader):

        if data_type == 'valid' and step >= args.max_val_batches:
            break

        image = batch["rgb"].to(dtype=weight_dtype)
        if args.target_model == 'xtc_depth': # ty: xtc model was trained with 256x256 images
            image = tvf.resize(image, 256, PIL.Image.BILINEAR)
            gt = tvf.resize(batch['gt'], 256, PIL.Image.BILINEAR)
        elif args.target_model in ['supermodel_depth','dpt_depth']: # ty: supermodel was trained with 384x384 images
            gt = tvf.resize(batch['gt'], 384, PIL.Image.BILINEAR)
        
        preds = rgb2depth(image.to(accelerator.device))
        
        if args.loss == 'l1':
            loss = -((preds - gt) * batch['mask']).abs().mean()  # l1 loss
        elif args.loss == 'midas':
            _, loss = midas_loss_(preds.float(), gt[:,:1], batch['mask'].bool())  # scale shift invariant loss
            loss = -1. * loss
        else:
            raise ValueError(f'loss {args.loss} is not supported.')
        losses.append(loss.detach().item())
        

        progress_bar.set_description(f'Initial validation. Steps: {step} / {len(dataloader)}')
    torch.cuda.empty_cache()
    return losses

if __name__ == "__main__":
    main()
