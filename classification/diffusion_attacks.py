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
import warnings
from pathlib import Path
from typing import Optional
import inspect
from collections import Counter

import json
import glob
import yaml

import numpy as np
import pandas as pd
import PIL

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModel

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm


import diffusers
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import torchvision.transforms as tf
import torchvision.transforms.functional as tvf
import torchvision.utils as tvu

import pickle
import cv2 as cv
import pdb

from utils import *
from functools import partial
from data.waterbirds import WaterbirdsDataset
from data.iwilds import IWildCamDataset
from sd_pipeline import img2img_pipeline, textual_inversion_loss


if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__)


def save_progress(text_encoder, placeholder_token_ids, placeholder_token_names, accelerator, save_path):
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
    learned_embeds_dict = {}
    for token, token_id in zip(placeholder_token_names, placeholder_token_ids):
        learned_embeds_dict[token] = learned_embeds[token_id].detach().cpu()

    torch.save(learned_embeds_dict, save_path)


def parse_args():

    config_parser = argparse.ArgumentParser(description="Training Config")
    config_parser.add_argument('-c', '--config', default=None, type=str, metavar='FILE', nargs="*",
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=False,
        help="Save only the embeddings for the new concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
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
        "--train_data_dir", type=str, default=None, help="The folder containing the training data."
    )
    parser.add_argument(
        "--masks_data_dir", type=str, default=None, help="The folder containing the instance masks data."
    )
    # parser.add_argument(
    #     "--conditioning_data_dir", type=str, default=None, required=True, help="A folder containing the conditioning training data."
    # )
    # parser.add_argument(
    #     "--building", type=str, default=None, required=True, help="Name of taskonomy building used for training."
    # )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=False,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--num_new_tokens",
        type=int,
        default=1,
        help="Number of new tokens to add to the tokenizer.",
    )
    parser.add_argument(
        "--placeholder_token_init_mode",
        type=str,
        default="mean_emb_cov_emb",
        help="How to initialize the new tokens.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, help="A token to use as initializer word."
    )
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
        "--dataset", type=str, default="iwilds", help="Dataset to use for training."
    )

    parser.add_argument(
        "--model", type=str, default="resnet50", help="Which model to get feedback from"
    )
    parser.add_argument(
        "--num_images", type=int, default=None, help="Number of images to use for training."
    )
    parser.add_argument(
        '--use_classes', default=[0], type=int, nargs='+',
    )
    parser.add_argument(
        '--use_places', default=[0], type=int, nargs='+',
    )
    parser.add_argument(
        "--strength", type=float, default=0.5, help="How much noise to apply to initial image."
    )
    parser.add_argument(
        "--daylight_time", action="store_true", help="only optimize on images from day time."
    )
    parser.add_argument(
        "--night_time", action="store_true", help="only optimize on images from night time."
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
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_baseline_prompt",
        type=str,
        default=None,
        help="A baseline prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="comic,cartoon,synthetic,rendered,animated,painting,sketch,drawing,highly saturated,humans,people",
        help="A negative prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
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
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=-10000000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument('--paste_masked_pixels', default=False, action='store_true', help='Whether to paste masked pixels from the original image.')
    parser.add_argument('--wandb_group', default='', type=str, help='Wandb group name.')
    parser.add_argument('--wandb_name', default='', type=str, help='Wandb run name.')
    parser.add_argument("--wandb_entity", default="", type=str, help="Wandb entity")


    parser.add_argument('--clip_text_guidance', default="", type=str)
    parser.add_argument('--clip_text_guidance_coef', default=1., type=float)
    parser.add_argument('--clip_image_guidance', default="", type=str)
    parser.add_argument('--clip_image_guidance_coef', default=1., type=float)


    parser.add_argument('--ti_guidance_coef', default=0., type=float)
    parser.add_argument('--ti_guidance_num_samples', default=None, type=int)

    parser.add_argument('--skip_adv_loss_steps', default=0, type=int)

    parser.add_argument("--keep_classes_iwilds", default=None, nargs="*", help="If specified, keep only the listed classes in iwilds")

    parser.add_argument("--keep_location_iwilds", default=None, type=int, nargs="*", help="If specified, keep only the listed locations in iwilds")

    

    # parser.add_argument("--no_target_is_background", dest="target_is_background", action="store_false", default=True)
    parser.add_argument("--loss_type", default='background', type=str, help='Loss type for adversarial loss.')

    parser.add_argument("--good_logging", default=False, action="store_true", help="If true, do automatically intelligent logging")

    parser.add_argument("--no_adv_opt", action="store_true", default=False)

    parser.add_argument("--return_image_before_paste", action="store_true", default=False)

    parser.add_argument("--no_inpainting", action="store_false", dest="use_mask_inpainting", default=True)

    args_config, remaining = config_parser.parse_known_args()

    if args_config.config:
        for config_file in args_config.config:
            with open(config_file, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)
                
    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # if args.conditioning_data_dir is None:
    #     raise ValueError("You must specify a train data directory.")

    return args



#prompt_templates_small = [
#    "{}, photorealistic, photo, highly detailed",
#    "{}, bright, photorealistic, photo, highly detailed",
#    "{}, dark, photorealistic, photo, highly detailed",
#]

prompt_templates_small = ["{}"]



def read_image(path):
  # with tf.io.gfile.GFile(path, 'rb') as f:
  #   return np.array(Image.open(f))
    return np.array(Image.open(path))






def image_name_to_id(name):
  return name.rstrip('.jpg')



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

    ## NAMING

    if args.good_logging:

        new_name = args.wandb_name
        new_name+= f"_seed={args.seed}"
        new_name += f"_s={args.strength}"
        new_name += f"_num-tok={args.num_new_tokens}"
        new_name += f"_num-steps={args.num_inference_steps}"

        if args.daylight_time:
            new_name += "_daylight"

        if args.night_time:
            new_name += "_nighttime"

        if args.keep_classes_iwilds is not None:
            new_name += f"_only={args.keep_classes_iwilds}"


        if args.num_images is not None:
            new_name+= f"_n_train_imgs={args.num_images}"

        if args.clip_text_guidance:
            loss_type = args.clip_text_guidance.split(':', 1)[0]

            new_name += f"_clip-text-{loss_type}-coef{args.clip_text_guidance_coef}"

        if args.clip_image_guidance:
            loss_type = args.clip_image_guidance.split(':', 1)[0]
            new_name += f"_clip-image-{loss_type}-coef{args.clip_image_guidance_coef}"   


        if args.ti_guidance_coef > 0:
            new_name += f"_ti_guidance-{args.ti_guidance_coef}-coef"
            if args.ti_guidance_num_samples is not None:
                new_name += f"_n_ti_imgs={args.ti_guidance_num_samples}"
            

        args.wandb_name = new_name

        args.output_dir = os.path.join(args.output_dir, args.wandb_group, args.wandb_name)
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"args.json")
        with open(save_path,"w") as f:
            json.dump(vars(args), f, indent=4)


    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

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
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
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
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

    if args.clip_image_guidance:
        pretrained_model_name_or_path= "openai/clip-vit-large-patch14"
        perceptual_loss_model = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path)
        #perceptual_loss_model.to(accelerator.device, dtype=weight_dtype)
        #perceptual_loss_model.eval()
        perceptual_loss_model.requires_grad_(False)

    # load pretrained classifier model
    if args.dataset == "iwilds":
        from models.iwilds_models import IWildsAdversarialLoss, crop_wilds
        if args.model == "resnet50":
            from torchvision.models import resnet50
            ## loading alia checkpoints
            model = resnet50(weights=None, num_classes=7)
            path = "pretrained_models/classification/iwilds_alia_resnet50.pth"
            saved_state_dict = torch.load(path)
            model.load_state_dict({k.replace('module.', ''): v for k, v in saved_state_dict["net"].items()})
        elif args.model == "vit_b_16":
            from torchvision.models import vit_b_16
            model = vit_b_16(weights=None, num_classes=7)
            path = "pretrained_models/classification/iwilds_vit.pth"
            saved_state_dict = torch.load(path)
            saved_state_dict = {k.replace('module.', ''): v for k, v in saved_state_dict["net"].items()}
            for name in ["weight","bias"]:
                saved_state_dict[f"heads.head.{name}"] = saved_state_dict[f"heads.{name}"]
                del saved_state_dict[f"heads.{name}"]
            model.load_state_dict(saved_state_dict)

        else:
            raise NotImplementedError(f"{args.model} not implemented")

        model.eval()
        loss_module = IWildsAdversarialLoss(resolution=args.resolution, type=args.loss_type)
        crop_out_watermarks = transforms.Lambda(partial(crop_wilds, resolution=args.resolution))

    elif args.dataset == "waterbirds":
        from models.waterbirds_models import resnet50, WaterbirdsAdversarialLoss
        model = resnet50(num_classes=2)
        ckpt = torch.load("pretrained_models/classification/waterbirds.pth")
        ckpt['net'] = {k.replace('module.', ''): v for k, v in ckpt['net'].items()}
        model.load_state_dict(ckpt['net'])
        model.eval()
        loss_module = WaterbirdsAdversarialLoss()

    # Add the placeholder token in tokenizer
    placeholder_token_ids, placeholder_token_names = add_new_tokens(
        tokenizer,
        text_encoder,
        args.placeholder_token,
        args.num_new_tokens,
        args.placeholder_token_init_mode,
        init_token=args.initializer_token,
    )

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # controlnet.requires_grad_(False)
    model.requires_grad_(False)
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

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    logger.info(f"Learning rate set to {args.learning_rate}")

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    if args.dataset == "iwilds":
        train_dataset = IWildCamDataset(
            # cond_data_root=args.conditioning_data_dir,
            data_root=args.train_data_dir,
            masks_dir=args.masks_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
            # pred_size=384 if args.target_model == 'supermodel_depth' else 256,
            placeholder_token=','.join(placeholder_token_names), 
            center_crop=args.center_crop,
            split="train",
            negative_prompt=args.negative_prompt,
            daylight_time=args.daylight_time,
            night_time=args.night_time,
            keep_classes=args.keep_classes_iwilds
        )

        if args.num_images is not None:
            ## doing class-balanced random sampling

            labels = np.array(train_dataset.labels)
            num_present_classes = len(np.unique(labels))
            per_class_num_samples = int(args.num_images / num_present_classes)
            indicies = np.array(list(range(len(train_dataset))))
            final_indicies=[]
            for curr_class in np.unique(labels).tolist():
                curr_indicies = indicies[labels == curr_class]
                sampled_indicies = np.random.choice(curr_indicies, per_class_num_samples, replace=False)
                final_indicies += sampled_indicies.tolist()

            #indicies = np.random.choice(len(train_dataset), args.num_images, replace=False)
            train_dataset = torch.utils.data.Subset(train_dataset, final_indicies)


    elif args.dataset == "waterbirds":
        train_dataset = WaterbirdsDataset(
            data_root=args.train_data_dir,
            size=args.resolution,
            tokenizer=tokenizer,
            placeholder_token=','.join(placeholder_token_names),
            negative_prompt=args.negative_prompt,
            classes=args.use_classes,
            num_samples=args.num_images,
            places=args.use_places,
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, 
        # HACK: i still don't quite understand what accelerator does with the dataloader,
        # but this helps get the same bs at each iteration irrespective of the number of gpus
        drop_last=(accelerator.num_processes == 1),
    )
    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        # TODO: fix the hack below
        # HACK: we need to have *accelerator.num_processes, otherwise it goes to 0 earlier, idk why it does this
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
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
    # controlnet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    model.to(accelerator.device, dtype=weight_dtype)

    if args.clip_image_guidance:
        perceptual_loss_model.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and args.report_to == "wandb":
        accelerator.init_trackers(
            f"diffusion-attacks-{args.dataset}",
            config=vars(args),
            init_kwargs={"wandb":{
                "name": args.wandb_name,
                'group': args.wandb_group,
                "entity": args.wandb_entity,
        }})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    do_classifier_free_guidance = args.guidance_scale > 1.0
    
    def get_timesteps(num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = noise_scheduler.timesteps[t_start * noise_scheduler.order :]

        return timesteps, num_inference_steps - t_start

    strength = args.strength
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device)
    timesteps, num_inference_steps = get_timesteps(
        num_inference_steps=args.num_inference_steps, strength=strength, device=accelerator.device
    )
    logger.info(f"Using {num_inference_steps} inference steps with strength {strength}")
    # print("timesteps!!!!", timesteps)
    num_images_per_prompt = 1




    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    eta = 1.
    # torch.autograd.set_detect_anomaly(True)

    if args.clip_text_guidance:
        path = args.clip_text_guidance.split(':', 1)[1]
        clip_text_guidance_emb = torch.load(path, map_location=accelerator.device)
        if isinstance(clip_text_guidance_emb, list):
            clip_text_guidance_scale = 1. / torch.clamp_min(clip_text_guidance_emb[1], 1e-3)
            clip_text_guidance_scale = clip_text_guidance_scale.to(dtype=weight_dtype)[None]

            clip_text_guidance_emb = clip_text_guidance_emb[0].to(dtype=weight_dtype)[None]
        else:
            clip_text_guidance_scale = 1.

    if args.clip_image_guidance:
        path = args.clip_image_guidance.split(':', 1)[1]
        mean_target_clip_features = torch.load(path, map_location=accelerator.device)["mean"].to(dtype=weight_dtype)[None]
        std_target_clip_features = torch.load(path, map_location=accelerator.device)["std"].to(dtype=weight_dtype)[None]


    ## TI GUIDANCE DATASET CREATION

    if args.ti_guidance_coef > 0:
        if args.dataset == "waterbirds":
            ti_dataset = WaterbirdsDataset(
                data_root=args.train_data_dir,
                size=args.resolution,
                tokenizer=tokenizer,
                placeholder_token=','.join(placeholder_token_names),
                negative_prompt=args.negative_prompt,
                num_samples=args.ti_guidance_num_samples,
                split="val",
                random_state=args.seed,
            )
        elif args.dataset == "iwilds":
            ti_dataset = IWildCamDataset(
                # data_root=args.train_data_dir,
                # cond_data_root=args.conditioning_data_dir,
                data_root=args.train_data_dir,
                masks_dir=args.masks_data_dir,
                tokenizer=tokenizer,
                size=args.resolution,
                # pred_size=384 if args.target_model == 'supermodel_depth' else 256,
                placeholder_token='a photo of an animal in' + ','.join(placeholder_token_names), 
                center_crop=args.center_crop,
                split="test",
                negative_prompt=args.negative_prompt,
                daylight_time=args.daylight_time,
                night_time=args.night_time,
                keep_classes=args.keep_classes_iwilds,
                keep_location=args.keep_location_iwilds
            )

            if args.ti_guidance_num_samples is not None:

                indicies = np.array(list(range(len(ti_dataset))))
                final_indicies = np.random.choice(indicies, args.ti_guidance_num_samples, replace=False)

                #indicies = np.random.choice(len(train_dataset), args.num_images, replace=False)
                ti_dataset = torch.utils.data.Subset(ti_dataset, final_indicies)


        ti_loader = torch.utils.data.DataLoader(
            ti_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True,
        )
        ti_iter = iter(ti_loader)


    for epoch in range(first_epoch, args.num_train_epochs):

        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
                
            logs = {}
            if len(batch['conditioning'].shape) == 3:
                batch['conditioning'] = batch['conditioning'][:, None]
        
            with accelerator.accumulate(text_encoder):
                loss = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
                
                prompt_embeds = None
                if args.clip_text_guidance:
                    # just to get prompt_embeds here, otherwise it's the same as in the img2img_pipeline
                    prompt_embeds = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
                    
                    if args.clip_text_guidance.startswith('l2'):
                        text_guidance_loss = ((prompt_embeds - clip_text_guidance_emb).pow(2) * clip_text_guidance_scale).mean()
                    elif args.clip_text_guidance.startswith('cosine'):
                        text_guidance_loss = 1 - F.cosine_similarity(prompt_embeds, clip_text_guidance_emb).mean()

                    logs['text_guidance_loss'] = text_guidance_loss.detach().item()

                    loss = loss + args.clip_text_guidance_coef * text_guidance_loss

                #with torch.set_grad_enabled(global_step >= args.skip_adv_loss_steps):
                image = img2img_pipeline(
                    noise_scheduler,
                    vae,
                    unet,
                    text_encoder,

                    batch,
                    len(batch["pixel_values"]),

                    args.resolution,
                    strength,
                    do_classifier_free_guidance,
                    timesteps,
                    args.guidance_scale,
                    eta,
                    args.paste_masked_pixels,

                    weight_dtype,
                    accelerator.device,
                    generator,

                    prompt_embeds=prompt_embeds,
                    return_image_before_paste=args.return_image_before_paste,
                    use_mask_inpainting=args.use_mask_inpainting
                )

                if args.return_image_before_paste:
                    image, image_before_paste = image

                adv_loss = loss_module.get_loss(model=model, gen_images=image, batch=batch)

                if global_step >= args.skip_adv_loss_steps and not args.no_adv_opt:
                    loss = loss + adv_loss


                if args.clip_image_guidance:

                    target_image = image_before_paste if args.return_image_before_paste else image

                    clip_processed_image = preprocess_clip(target_image).to(accelerator.device, dtype=weight_dtype)
                    clip_features = perceptual_loss_model(clip_processed_image).pooler_output
                    
                    if args.clip_image_guidance.startswith('l2'):
                        perceptual_loss =((mean_target_clip_features - clip_features).pow(2) / std_target_clip_features).mean()

                    elif args.clip_image_guidance.startswith('cosine'):
                        perceptual_loss = 1 - F.cosine_similarity(mean_target_clip_features, clip_features).mean()

                    logs['image_guidance_loss'] = perceptual_loss.detach().item()
                    loss +=  args.clip_image_guidance_coef * perceptual_loss


                if args.ti_guidance_coef > 0:
                    try:
                        ti_batch = next(ti_iter)
                    except StopIteration:
                        ti_iter = iter(ti_loader)
                        ti_batch = next(ti_iter)

                    ti_batch = {k: v.to(accelerator.device) for k, v in ti_batch.items() if isinstance(v, torch.Tensor)}

                    ti_loss = textual_inversion_loss(
                        vae,
                        text_encoder,
                        noise_scheduler,
                        unet,

                        ti_batch,
                        weight_dtype,
                        # guidance_scale=args.guidance_scale,
                    )
                    logs['ti_guidance/loss'] = ti_loss.detach().item()

                    loss = loss + args.ti_guidance_coef * ti_loss


                # pdb.set_trace()
                accelerator.backward(loss)
                # grad_norm = text_encoder.get_input_embeddings().weight.grad[placeholder_token_id].norm()
                grad_norm = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.grad[placeholder_token_ids].norm().item()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.arange(0, 49408)
                assert min(placeholder_token_ids) >= 49408, "Smt went wrong with the placeholder token ids"
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_ids, placeholder_token_names, accelerator, save_path)

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")



            with torch.no_grad():
                metrics = loss_module.get_eval_metrics(model=model, gen_images=image, batch=batch)

            logs.update({
                "loss": loss.detach().item(),
                "adv_loss": adv_loss.detach().item(), 
                "grad_norm": grad_norm,
                "lr": lr_scheduler.get_last_lr()[0],
                **metrics,
            })

            if global_step % args.validation_steps == 0 or global_step == 1:
                orig_gen_images = torch.cat([0.5*(1.+batch["pixel_values"]), image], dim=0)
                if args.dataset == "iwilds":
                    orig_gen_images = crop_out_watermarks(orig_gen_images)
                orig_gen_images = tvu.make_grid(orig_gen_images,nrow=args.train_batch_size)
                # caption = ' | '.join(batch['prompt'])
                caption = ''
                logs['train'] = wandb.Image(orig_gen_images, caption=caption)

                # plot the images with the mask
                orig_images = 0.5*(1.+batch["pixel_values"])
                orig_images[batch['conditioning'][:, 0].cpu() == 0, 2] = 0.9
                adv_images = image
                adv_images[batch['conditioning'][:, 0].cpu() == 0, 2] = 0.9
                orig_gen_images = torch.cat([orig_images, adv_images], dim=0)
                if args.dataset == "iwilds":
                    orig_gen_images = crop_out_watermarks(orig_gen_images)
                    
                orig_gen_images = tvu.make_grid(orig_gen_images, nrow=args.train_batch_size)
                logs['train_mask'] = wandb.Image(orig_gen_images, caption=caption)


                if args.ti_guidance_coef > 0:
                    logs['ti_guidance/images'] = wandb.Image(tvu.make_grid(ti_batch['pixel_values'], nrow=args.train_batch_size))

            progress_bar.set_postfix(**logs)
            
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub and args.only_save_embeds:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = not args.only_save_embeds
        if save_full_model:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        save_path = os.path.join(args.output_dir, "learned_embeds.bin")
        save_progress(text_encoder, placeholder_token_ids, placeholder_token_names, accelerator, save_path)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    # accelerator.end_training()


if __name__ == "__main__":
    main()
