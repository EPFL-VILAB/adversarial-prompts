python3 generate_adv_images.py \
    --images-path /scratch/alekseev/work/controlling-controlnet/taskonomy_splits/multi-iter-paths/50k_gen_15k_train/gen_1_iter.txt \
    --save-root /scratch/alekseev/taskonomy_normalized/test/ \
    --num-inference-steps 15 \
    --seed 1 \
    --batch-size 3 \
    --prompt None \
    --gpu-id 0 --num-gpus 1 \
    --sampling mean_emb_cov_emb --num-new-tokens 123
    # --adversarial --tokens-root /scratch/alekseev/work/experiments/diff_dataset \
    # --adversarial-runs-file /scratch/alekseev/work/experiments/finetune_sd/adv-tokens/multi-8runs-t07-501-1_iter-runs.json \

python3 generate_adv_images.py \
    --images-path /scratch/alekseev/work/controlling-controlnet/taskonomy_splits/multi-iter-paths/50k_gen_15k_train/gen_1_iter.txt \
    --save-root /scratch/alekseev/taskonomy_normalized/test/ \
    --num-inference-steps 15 \
    --seed 1 \
    --batch-size 3 \
    --prompt None \
    --gpu-id 0 --num-gpus 1 \
    --sampling mean_emb_cov_emb --num-new-tokens 123
    # --adversarial --tokens-root /scratch/alekseev/work/experiments/diff_dataset \
    # --adversarial-runs-file /scratch/alekseev/work/experiments/finetune_sd/adv-tokens/multi-8runs-t07-501-1_iter-runs.json \