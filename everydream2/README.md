# EveryDream2 Docker Container

This folder contains Dockerfiles for EveryDream.

## Sample Docker run command

```
docker run -it --rm --gpus "device=0" \
-v /mnt/user/work/models/Stable-diffusion/:/sdmodels \
-v /mnt/user/work/everydream/:/data \
entmike/everydream2 bash
```

## Convert checkpoint

```
python3 utils/convert_original_stable_diffusion_to_diffusers.py --scheduler_type ddim \
--original_config_file v2-inference-v.yaml \
--image_size 768 \
--checkpoint_path /sdmodels/v2-1_768-nonema-pruned.ckpt \
--prediction_type v_prediction \
--upcast_attn False \
--dump_path "/data/ckpt_cache/v2-1_768-nonema-pruned"
```

## Sample training command

```
python3 train.py  --ckpt_every_n_minutes 10 --useadam8bit \
--resume_ckpt /data/ckpt_cache/v2-1_768-nonema-pruned \
--max_epochs 25 --data_root /data/training_data/family \
--lr_scheduler cosine --project_name sampleproject --batch_size 6 \
--sample_steps 200 --lr 3e-6
```