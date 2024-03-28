import inspect
from utils import prepare_mask_and_masked_image, randn_tensor
import torch
import torch.nn.functional as F

def img2img_pipeline(
        noise_scheduler,
        vae,
        unet,
        text_encoder,
        batch,
        train_batch_size,
        resolution,
        strength,
        do_classifier_free_guidance,
        timesteps,
        guidance_scale,
        eta,
        paste_masked_pixels,
        weight_dtype,
        device,
        generator,
        prompt_embeds=None,
        return_image_before_paste=False,
        use_mask_inpainting=True
    ):

    def prepare_mask_latents(
        mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        
        mask = torch.nn.functional.interpolate(
            mask, size=(height // vae_scale_factor, width // vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)
        masked_image_latents = _encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents


    def prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)
            image_latents = _encode_vae_image(image=image, generator=generator)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else noise_scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * noise_scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * noise_scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs


    def prepare_extra_step_kwargs(generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(noise_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(noise_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs


    def _encode_vae_image(image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                vae.encode(image[i : i + 1]).latent_dist.sample(generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = vae.encode(image).latent_dist.sample(generator=generator)

        image_latents = vae.config.scaling_factor * image_latents

        return image_latents

    extra_step_kwargs = {}
    extra_step_kwargs['generator'] = generator
    # print("timesteps!!!!", timesteps)
    num_images_per_prompt = 1
    latent_timestep = timesteps[:1].repeat(train_batch_size * num_images_per_prompt)
    # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
    is_strength_max = strength == 1.0
    # timesteps = noise_scheduler.timesteps
    # num_warmup_steps = len(timesteps) - num_inference_steps * noise_scheduler.order
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    if prompt_embeds is None:
        prompt_embeds = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

    negative_prompt_embeds = text_encoder(batch["neg_input_ids"])[0].to(dtype=weight_dtype)
    if do_classifier_free_guidance:
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 5. Preprocess mask and image
    mask, masked_image, init_image = prepare_mask_and_masked_image(
        batch["pixel_values"], batch["conditioning"], resolution, resolution, return_image=True
    )
    # breakpoint()
    mask_condition = mask.clone()

    # 6. Prepare latent variables
    num_channels_latents = vae.config.latent_channels
    num_channels_unet = unet.config.in_channels
    return_image_latents = num_channels_unet == 4

    latents_outputs = prepare_latents(
        train_batch_size * num_images_per_prompt,
        num_channels_latents,
        resolution,
        resolution,
        prompt_embeds.dtype,
        device,
        generator,
        None,
        image=init_image,
        timestep=latent_timestep,
        is_strength_max=is_strength_max,
        return_noise=True,
        return_image_latents=return_image_latents,
    )

    if return_image_latents:
        latents, noise, image_latents = latents_outputs
    else:
        latents, noise = latents_outputs

    if use_mask_inpainting:

        mask, masked_image_latents = prepare_mask_latents(
            mask,
            masked_image,
            train_batch_size * num_images_per_prompt,
            resolution,
            resolution,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

    extra_step_kwargs = prepare_extra_step_kwargs(generator, eta)

    # 10. Denoising loop
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        # concat latents, mask, masked_image_latents in the channel dimension
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

        if  use_mask_inpainting and (num_channels_unet == 9):
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

        # predict the noise residual
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            # cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        
        if use_mask_inpainting and (num_channels_unet == 4):
            init_latents_proper = image_latents  #[:1]
            init_mask = mask[:train_batch_size] if do_classifier_free_guidance else mask

            if i < len(timesteps) - 1:
                noise_timestep = timesteps[i + 1]
                init_latents_proper = noise_scheduler.add_noise(
                    init_latents_proper, noise, torch.tensor([noise_timestep])
                )

            latents = (1 - init_mask) * init_latents_proper + init_mask * latents


    # 8. Post-processing
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1) #.float()

    if return_image_before_paste:
        before_paste_image = image

    if paste_masked_pixels:
        mask_condition = mask_condition.to(image.device, dtype=image.dtype)
        init_image = init_image.to(image.device, dtype=image.dtype)
        # sclae init_image to [-1, 1]
        image = image * mask_condition + (init_image / 2 + 0.5) * (1 - mask_condition)

    if return_image_before_paste:
        return image, before_paste_image

    return image



def textual_inversion_loss(
        vae,
        text_encoder,
        noise_scheduler,
        unet,

        batch,
        weight_dtype,
    ):
    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
    latents = latents * vae.config.scaling_factor

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

    # Predict the noise residual
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
    
    # ignore the masked pixels in the loss
    # mask = batch["conditioning"].to(dtype=weight_dtype)

    # height, width = mask.shape[-2:]
    # vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    # mask = torch.nn.functional.interpolate(
    #     mask, size=(height // vae_scale_factor, width // vae_scale_factor)
    # )

    # model_pred = model_pred * mask
    # target = target * mask


    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss