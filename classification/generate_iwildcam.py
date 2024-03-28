import torch
import numpy as np
import torch
import argparse
import os
import PIL
from tqdm import tqdm
import logging
from argparse import Namespace
import json

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from sd_pipeline import img2img_pipeline
from utils import add_new_tokens
from data.iwilds import IWildCamDataset
from pathlib import Path
import json

alia_prompts = ['a photo of {class_name} in a grassy field with trees and bushes',
 'a photo of {class_name} in a forest in the dark',
 'a photo of {class_name} near a large body of water in the middle of a field',
 'a photo of {class_name} walking on a dirt trail with twigs and branches',
 'a camera trap photo of a {class_name}']

def main(args,args_config):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # Load the adversarial tokens
    if args_config.adversarial_token_path:
        _, placeholder_token_names = add_new_tokens(
            tokenizer,
            text_encoder,
            args.placeholder_token,
            args.num_new_tokens,
            'file',
            embs_path=args_config.adversarial_token_path,
        )
    else:
        placeholder_token_names = [args.placeholder_token]

    strength = args.strength
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)

    def get_timesteps(num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = noise_scheduler.timesteps[t_start * noise_scheduler.order :]

        return timesteps, num_inference_steps - t_start

    timesteps, num_inference_steps = get_timesteps(
        num_inference_steps=args.num_inference_steps, strength=strength, device=device
    )
    do_classifier_free_guidance = True
    eta = 1.0

    if args.use_alia_prompt_idx != -1:

        prompt_templates = [alia_prompts[args.use_alia_prompt_idx]]

        train_dataset = IWildCamDataset(
            tokenizer=tokenizer,
            size=args.resolution,
            placeholder_token="", 
            split="train",
            negative_prompt=args.negative_prompt,
            prompt_templates=prompt_templates,
            use_bbox=args.use_bbox,
            daylight_time=args.daylight_time,
            night_time=args.night_time,
        )

    else:

        train_dataset = IWildCamDataset(
            tokenizer=tokenizer,
            size=args.resolution,
            placeholder_token=','.join(placeholder_token_names), 
            split="train",
            negative_prompt=args.negative_prompt,
            use_bbox=args.use_bbox,
            daylight_time=args.daylight_time,
            night_time=args.night_time,
        )

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    mapping_dict={}

    for batch_idx, batch in enumerate(tqdm(train_dataloader)):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        for gen_idx in range(args.gen_per_image):
            if not args.mapping_only:
                with torch.no_grad():
                    image = img2img_pipeline(
                        noise_scheduler,
                        vae,
                        unet,
                        text_encoder,
                        batch,
                        batch['pixel_values'].shape[0],
                        args.resolution,
                        strength,
                        do_classifier_free_guidance,
                        timesteps,
                        args.guidance_scale,
                        eta,
                        args.paste_masked_pixels,
                        weight_dtype,
                        device,
                        generator,
                    )

                for i in range(image.shape[0]):
                    img_idx = batch_idx * args.batch_size + i
                    img = PIL.Image.fromarray((image[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

                    c= train_dataset.class_names[batch['labels'][i]]
                    directory = os.path.join(args.save_dir, c)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    path = f'{directory}/{img_idx}-{gen_idx}.png'

                    ## saving mapping
                    orig_file_path = batch["file_path"][i]
                    mapping_dict[f"{c}/{img_idx}-{gen_idx}"] = orig_file_path
                    #assert not os.path.exists(path), f'Path {path} already exists!'
                    img.save(path)
            else:
                num_images = batch['pixel_values'].shape[0]

                for i in range(num_images):
                    img_idx = batch_idx * args.batch_size + i
                    #img = PIL.Image.fromarray((image[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

                    c= train_dataset.class_names[batch['labels'][i]]
                    directory = os.path.join(args.save_dir, c)
                    #if not os.path.exists(directory):
                    #    os.makedirs(directory)
                    path = f'{directory}/{img_idx}-{gen_idx}.png'
                    orig_file_path = batch["file_path"][i]
                    mapping_dict[f"{c}/{img_idx}-{gen_idx}"] = orig_file_path
                    #assert not os.path.exists(path), f'Path {path} already exists!'
                    #img.save(path)

    save_path = os.path.join(args.save_dir, "mapping.json")
    with open(save_path, "w") as f:
        json.dump(mapping_dict,f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./iwilds_generated_data')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--strength', type=float, default=1.)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_inference_steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--num_new_tokens",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="comic,cartoon,synthetic,rendered,animated,painting,sketch,drawing,highly saturated,humans,people",
        help="A negative prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_images", type=int, default=None, help="Number of images to use for training."
    )
    parser.add_argument(
        "--gen_per_image", type=int, default=1
    )
    parser.add_argument(
        '--resolution', default=384, type=int,
    )
    parser.add_argument(
        '--batch_size', default=32, type=int,
    )
    parser.add_argument(
        '--dataloader_num_workers', default=4, type=int,
    )
    parser.add_argument(
        '--adversarial_token_path', type=str, default="",
    )
    parser.add_argument(
        '--paste_masked_pixels', action='store_true', default=False,
    )

    parser.add_argument(
        "--daylight_time", action="store_true", default=False
    )

    parser.add_argument(
        "--night_time", action="store_true", default=False
    )

    parser.add_argument(
        "--no_daylight_time", action="store_false", dest="daylight_time"
    )

    parser.add_argument(
        "--no_night_time", action="store_false", dest="night_time"
    )

    parser.add_argument(
        "--mapping_only", action="store_true", default=False
    )


    parser.add_argument("--use_bbox", action="store_true", default=False)

    parser.add_argument("--use_alia_prompt_idx", default=-1, type=int)

    #parser.add_argument("--path_to_args_file", default=None)

    config_parser = argparse.ArgumentParser(description="Training Config")
    config_parser.add_argument('--adversarial_token_path', default=None, type=str, metavar='FILE')


    args_config, remaining = config_parser.parse_known_args()

    print(args_config.adversarial_token_path)

    if args_config.adversarial_token_path:
        args_path = os.path.join(Path(args_config.adversarial_token_path).parent, "args.json")

        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                override_args = json.load(f)

        else:
            raise ValueError(f"{args_path} doesn't exist")

        parser.set_defaults(**override_args)

        #orig_dict = vars(args)
        #orig_dict.update(override_args)
#
        ## the merged object
        #args = Namespace(**orig_dict)
    
    args = parser.parse_args(remaining)


    assert not(args.use_alia_prompt_idx != -1 and args_config.adversarial_token_path)

    if args.use_alia_prompt_idx != -1:
        curr_alia_prompt= alia_prompts[args.use_alia_prompt_idx]
        curr_alia_prompt = curr_alia_prompt.replace(" ", "_")
        args.save_dir = os.path.join(args.save_dir, curr_alia_prompt)

    elif args_config.adversarial_token_path is not None:
        ## getting wandb group and run name
        run_name = os.path.join(*Path(args_config.adversarial_token_path).parts[-3:-1])
        args.save_dir = os.path.join(args.save_dir, run_name)


    append="gen"
    append+= f"_seed={args.seed}"
    append+= f"_num-steps={args.num_inference_steps}"
    append += f"_s={args.strength}"
    append += f"_num-tok={args.num_new_tokens}"
    append += f"_guidance={args.guidance_scale}"

    if args.resolution != 384:
        append += f"_resolution={args.resolution}"

    args.save_dir = os.path.join(args.save_dir, append)



    main(args, args_config)