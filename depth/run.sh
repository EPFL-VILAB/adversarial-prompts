lr=5.0e-04

seed=0
num_new_tokens=4
CUDA_VISIBLE_DEVICES=0 python diffusion_attacks_image_guidance.py \
    --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
    --taskonomy_data_root /scratch/alekseev/data/taskonomydata/ \
    --mask_data_root /scratch/alekseev/data/taskonomymask/ \
    --rgb2depth_checkpoint_path /scratch/alekseev/work/rgb2depth_consistency.pth \
    --dataset taskonomy \
    --image_point 103 --image_view 0 --max_val_batches 100 \
    --loss l1 --building airport --building_val airport --resolution 384 --train_batch_size 1 --gradient_accumulation_steps 1 \
    --max_train_steps 2001 --learning_rate=$lr --scale_lr --lr_scheduler=constant --lr_warmup_steps=0 \
    --guidance_scale 7 --validation_steps 100 --dataloader_num_workers 8 \
    --num_validation_images 5 --num_inference_steps 5 \
    --mixed_precision fp16 --seed $seed --report_to wandb --save_steps 100 --target_model xtc_depth --num_new_tokens $num_new_tokens \
    --initializer_token room --pretrained_tokens_sample_mode 'mean_emb_cov_emb' \
    --early_stop --output_dir=debug \
    --data_paths_file /scratch/alekseev/work/controlling-controlnet/taskonomy_splits/multi-iter-paths/20iter_10k/train_1_iter.txt \
    --early_stop_threshold -0.2 \
    --taskonomy_split_path /scratch/alekseev/work/train_val_test_fullplus.csv \
    --target_domain_images_path_file /scratch/alekseev/work/controlling-controlnet/target_images/gaussian_blur_5.txt --clip_image_guidance --clip_image_guidance_coef 1. \
    # --clip_text_guidance l2:/scratch/teresa/diffdataset/controlling_controlnet_all/attack_granularity/controlling-controlnet/target_text_embeds/iso_noise_embed.pt
    