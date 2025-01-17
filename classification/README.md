# Classification 

[TOC]



## Pretrained models

For ease of use, we provide our pretrained model checkpoints, for Waterbirds and iWildCam, used during our experiments. Please run `./download.sh` to download the checkpoints.

## iWildCam

### (Guided) Adversarial optimization
Here is a sample script to run adversarial optimization for iWildCam classification:

```shell

 accelerate launch diffusion_attacks.py \
 --config configs/iwilds/iwilds.yml \
 --output_dir "iwilds_adv_opt" \
 --seed 5 \
 --num_new_tokens 10 \
 --num_inference_steps 5 \
 --learning_rate 1e-3 \
 --strength 0.8 \
 --guidance_scale 5.0 \
 --paste_masked_pixels \
 --max_train_steps 10000 

```



Where



To add image guidance, please add the following arguments:

```

  --clip_image_guidance cosine:<PATH_TO_CLIP_EMBEDDING> \
  --clip_image_guidance_coef 10
  --skip_adv_loss_steps 2000
```


To add text guidance, please add the following arguments:

```
  --clip_text_guidance l2:<PATH_TO_CLIP_EMBEDDING>
  --clip_text_guidance_coef 1
```
Where

- `<PATH_TO_CLIP_EMBEDDING>` points to the target CLIP embedding e.g. `"iwildcam_target_image_embeds/test_day_loc=287_clip_embeds.pt"` which is the mean CLIP embedding of 64 images from the test location 287 (location=1 in the paper) during daytime.
- To create the target CLIP embeddings for iWildCam, we provide the `clip_image_guidance.py` script.

Note that, in the `--clip_text_guidance` or `--clip_image_guidance`  argument, the first part before `:` can be either `l2` or `cosine`, specifying the type of guidance loss.



For iWildCam, you may wish to add the following arguments (depending on the target domain)
```
--daylight_time 
--loss_type uniform 
```

We provide the python script `generate_iwilds_image_embeds.py` to generate the 4 target domains for iWildCam and as a template to generate target CLIP embeddings. For ease of use, we provide the precomputed embeddings for iWildCam in `iwildcam_target_image_embeds`.

### Data generation
Here is a sample script to run data generation for iWildCam:


```shell
python generate_iwildcam.py \
--num_inference_steps 5 \
--resolution 384 
--guidance_scale 5 
--strength 0.8
```

To generate iWildCam using ALIA prompts, use the `--use_alia_prompt_idx $i` argument where $i is the corresponding prompt index.

To generate data with (guided) adversarial prompts, use the `--adversarial_token_path <PATH_TO_ADVERSARIAL_PROMPT>` argument where `<PATH_TO_ADVERSARIAL_PROMPT>` points to the adversarial prompt generated by `diffusion_attacks.py`. It should usually point to a file named "`<ROOT>/<RUN_NAME>/learned_embeds-<EPOCH>.bin`".


## Waterbirds

Running experimnets for Waterbirds is similar to iWildCam, with the difference that we find adversarial embeddings for each of two classes, and use the `--use_classes` argument to specify the class for which to generate adversarial embeddings. `0` corresponds to the class `landbird` and `1` corresponds to the class `waterbird`. We also set `--use_places` to the same value as `--use_classes` to use only the training data with the spurious correlation.

You can load the Waterbirds dataset with precomputed masks using the SAM model [here](https://drive.google.com/file/d/1UsY1f0T-cDnuE4kd2cqDHQxj0olc52HK/view?usp=sharing).

### (Guided) Adversarial optimization
Use the following template to find guided adversarial prompts (omit `--clip_text_guidance` to find just adversarial prompts):
```shell
  accelerate launch diffusion_attacks.py \
  --config configs/waterbirds/waterbirds.yml \
  --output_dir "waterbirds_adv_opt" \
  --num_new_tokens 5 \
  --use_classes <0 or 1> \
  --use_places <same as use_classes> \
  --clip_text_guidance cosine:waterbirds_text_guidance.pt
  --clip_text_guidance_coef 10 
  --placeholder_token <landbird for 0 or waterbird for 1>_adv_token 
```

### Data generation
Use the following template to generate data for Waterbirds:
```shell
  python generate.py \
  --adversarial_token_path <path to .bin embedding file> \
  --placeholder_token <waterbird or landbird>_adv_token \
  --use_classes <0 or 1> \
  --use_places <same as use_classes> \
  --save_dir <path to gen data> \
  --num_new_tokens 5
  --guidance_scale 7 \
  --batch_size 20 \
  --paste_masked_pixels \
  --num_images 1000
```


## Fine-tuning

To fine-tune the model on generated data, we use the [ALIA](https://github.com/lisadunlap/ALIA) repository.

First, install the repository:

```shell
git clone https://github.com/lisadunlap/ALIA.git
cd ALIA
conda env create -f environment.yaml
conda activate ALIA
pip install -e .
```

Here is a sample script to finetune on generated data, using a similar setting to our paper:

```shell
cd ALIA
python main.py \
--config=configs/iWildCamMini/generated.yaml \
--finetune
```
By default, it will finetune on 2224 data samples, you may use `data.num_extra=556` argument to finetune only 556 samples.

The config file `generated.yaml` will look like this:
```yaml
base_config: configs/iWildCamMini/base.yaml
name: iWildCam-generated ## wandb arguments, feel free to change the names
run_name: generated

epochs: 20 

data: 
  base_dataset: iWildCamMini
  base_root: "<path/to/iWildCam>"
  extra_dataset: BasicDataset
  extra_root: 
  - <path/to/data/generated/with/prompt1>
  -<path/to/data/generated/with/prompt2>
  - <path/to/data/generated/with/prompt3>
  - <path/to/data/generated/with/prompt4>
  extra_classes: ['background', 'cattle', 'elephants', 'impalas', 'zebras', 'giraffes', 'dik-diks']
  filter: false
```
**Disclaimer:** The original ALIA repository doesn't implement last-layer finetuning. However, this is very easy to add to `main.py`. The code to add will approximately look like this:
```python
if args.finetune:
    print("...finetuning")
    # freeze all bust last layer
    if args.model == "resnet50":
        for name, param in net.named_parameters(): 
            if 'fc' not in name:
                param.requires_grad = False
```
