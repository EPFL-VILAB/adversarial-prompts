## Depth 

### Download pretrained models

We perform adversarial optimization on the following pre-trained models: 
1. [UNet](https://arxiv.org/abs/1505.04597) model trained on the Taskonomy dataset with consistency constraints [[Download link](https://github.com/EPFL-VILAB/XTConsistency?tab=readme-ov-file#download-consistency-trained-models)]
2. [DPT](https://arxiv.org/abs/2103.13413) model trained on Omnidata. We used the V1 models. [[Download link](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch#download-pretrained-models)]. 


### (Guided) Adversarial optimization
Here is a sample script to run adversarial optimization for depth estimation on Taskomomy dataset:

```shell

python diffusion_attacks_image_guidance.py \
    --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
    --taskonomy_data_root <PATH_TO_TASKONOMY_IMAGES> \
    --mask_data_root <PATH_TO_TASKONOMY_IMAGE_MASKS> \
    --rgb2depth_checkpoint_path <PATH_TO_MODEL_CHECKPOINT> \
    --dataset taskonomy \
    --image_point 103 --image_view 0 --max_val_batches 100 \
    --loss <LOSS_FUNCTION> --building airport --building_val airport --resolution 384 --train_batch_size 4 --gradient_accumulation_steps 2 \
    --max_train_steps 500 --learning_rate 5.0e-4 --scale_lr --lr_scheduler constant --lr_warmup_steps 0 \
    --guidance_scale 7 --validation_steps 100 --dataloader_num_workers 8 \
    --num_validation_images 5 --num_inference_steps 5 \
    --mixed_precision fp16 --seed 42 --report_to wandb --save_steps 100 --target_model <PRETRAINED_MODEL_NAME> --num_new_tokens 4 \
    --initializer_token room \
    --early_stop --early_stop_threshold -0.2 \
    --output_dir <SAVE_DIR> \
    --data_paths_file <PATH_TO_FILE_CONTAINING_TRAIN_IMAGES> \
    --taskonomy_split_path <PATH_TO_TASKONOMY_TRAIN_VAL_SPLIT> \
```

Where

- `<PATH_TO_FILE_CONTAINING_TRAIN_IMAGES>` points to the text file that contains paths of training depth images.
- `<PATH_TO_TASKONOMY_TRAIN_VAL_SPLIT>` points to a `.csv` file that for every building specifies whether it's a train, validation or test building.
- `<PATH_TO_MODEL_CHECKPOINT>` points to the pre-trained model downloaded from [here](#download-pretrained-models).
- `<PRETRAINED_MODEL_NAME>` is `xtc_depth` for the pre-trained UNet model and `dpt_depth` for the DPT model.
- `<LOSS_FUNCTION>` is `l1` for the pre-trained UNet model and `midas` for the DPT model.

To add image guidance, please add the following arguments:

```
  --target_domain_images_path_file <PATH_TO_FILE_CONTAINING_TARGET_IMAGES> \
  --clip_image_guidance \
  --clip_image_guidance_coef 1
```

To add text guidance, please add the following arguments:

```
  --clip_text_guidance l2:<PATH_TO_CLIP_EMBEDDING>
  --clip_text_guidance_coef 1
```
Where

- `<PATH_TO_FILE_CONTAINING_TARGET_IMAGES>` points to the file containing target domain RGB images.
- `<PATH_TO_CLIP_EMBEDDING>` points to the target CLIP embedding.

Note that, in the `--clip_text_guidance`, the first part before `:` can be either `l2` or `cosine`, specifying the type of guidance loss.

Running the adversarial optimization on a 80GB A100 takes about 30mins (assuming no early stopping).

### Data generation
Here is a sample script to run data generation for depth estimation:

```shell

python generate_adv_images.py \
    --images-path <PATH_TO_FILE_CONTAINING_IMAGES> \
    --save-root <SAVE_DIR> \
    --num-inference-steps 15 \
    --seed 42 \
    --batch-size 16 \
    --gpu-id 0 --num-gpus 1 \
    --adversarial --tokens-root <PATH_TO_TOKENS_ROOT> \
    --adversarial-runs-file <PATH_TO_FILE_CONTAINING_TOKEN_INFO>
```

Where

- `<PATH_TO_FILE_CONTAINING_IMAGES>` points to the text file that contains paths of depth images for which to generate RGBs.
- `<PATH_TO_TOKENS_ROOT>` points to the parent directory for saved tokens.
- `<PATH_TO_FILE_CONTAINING_TOKEN_INFO>` points to the `.json` file of structure
  ```
  {<RUN_NAME>: [<EPOCH_1>, <EPOCH_2>, ...]
  ```
  The final token path should be i.e. `<PATH_TO_TOKENS_ROOT>/<RUN_NAME>/learned_embeds-<EPOCH_1>.bin`.

### Fine-tuning

To fine-tune the model on generated data, for the DPT model, please follow [Omnidata](https://github.com/EPFL-VILAB/omnidata/tree/main) repository.
