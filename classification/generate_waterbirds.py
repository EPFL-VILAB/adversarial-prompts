import torch
import numpy as np
import torch
import argparse
import os
import PIL
from tqdm import tqdm
import logging

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from sd_pipeline import img2img_pipeline
from data.waterbirds import WaterbirdsDataset
from utils import add_new_tokens


prompts = {
    "Cub2011": "a iNaturalist photo of a {} bird.",
    "iWildCamMini": "a camera trap photo of {} in the wild.",
    "Planes": "a photo of a {} airplane.",
}


def main(args):
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
    if args.adversarial_token_path:
        _, placeholder_token_names = add_new_tokens(
            tokenizer,
            text_encoder,
            args.placeholder_token,
            args.num_new_tokens,
            'file',
            embs_path=args.adversarial_token_path,
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

    if args.overwrite_prompt is None:
        train_dataset = WaterbirdsDataset(
            data_root='/scratch/atanov/data/',
            size=args.resolution,
            tokenizer=tokenizer,
            placeholder_token=','.join(placeholder_token_names),
            negative_prompt=args.negative_prompt,
            classes=args.use_classes,
            num_samples=args.num_images,
            places=args.use_places,
        )
    else:
        train_dataset = WaterbirdsDataset(
            data_root='/scratch/atanov/data/',
            size=args.resolution,
            tokenizer=tokenizer,
            placeholder_token='',
            negative_prompt=args.negative_prompt,
            classes=args.use_classes,
            num_samples=args.num_images,
            places=args.use_places,
            prompt_templates=[args.overwrite_prompt],
        )

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    for batch_idx, batch in enumerate(tqdm(train_dataloader)):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        for gen_idx in range(args.gen_per_image):
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
                c = ['Landbird', 'Waterbird'][batch['labels'][i]]
                directory = os.path.join(args.save_dir, c)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                path = f'{directory}/{img_idx}-{gen_idx}.png'
                assert not os.path.exists(path), f'Path {path} already exists!'
                img.save(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./diffusion_generated_data')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='waterbirds', choices=['waterbirds'])
    parser.add_argument('--strength', type=float, default=1.)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_inference_steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--overwrite_prompt', type=str, default=None)
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
        '--use_classes', default=[0], type=int, nargs='+',
    )
    parser.add_argument(
        '--use_places', default=[0], type=int, nargs='+',
    )
    parser.add_argument(
        '--resolution', default=384, type=int,
    )
    parser.add_argument(
        '--batch_size', default=2, type=int,
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
    args = parser.parse_args()

    main(args)