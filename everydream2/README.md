# EveryDream2 Docker Container

This folder contains Dockerfiles for EveryDream.

## Sample Docker run command

```
docker run -it --rm --gpus "device=0" \
-v /mnt/user/work/models/Stable-diffusion/:/sdmodels \
-v /mnt/user/work/everydream/:/data \
entmike/everydream2 bash
```

## Convert 2.1 checkpoint

```
python3 utils/convert_original_stable_diffusion_to_diffusers.py --scheduler_type ddim \
--original_config_file v2-inference-v.yaml \
--image_size 768 \
--checkpoint_path /sdmodels/v2-1_768-nonema-pruned.ckpt \
--prediction_type v_prediction \
--upcast_attn False \
--dump_path "/data/ckpt_cache/v2-1_768-nonema-pruned"
```

## Convert 1.x checkpoint

```
python3 utils/convert_original_stable_diffusion_to_diffusers.py --scheduler_type ddim \
--original_config_file v1-inference.yaml \
--image_size 512 \
--checkpoint_path /sdmodels/v1-5-pruned-emaonly.ckpt \
--prediction_type epsilon \
--upcast_attn False \
--dump_path "/data/ckpt_cache/v1-5-pruned-emaonly"
```

## Sample training command (2.x)

```
python3 train.py  --ckpt_every_n_minutes 10 --useadam8bit \
--resume_ckpt /data/ckpt_cache/v2-1_768-nonema-pruned \
--data_root /data/training_data/family \
--save_ckpt_dir /data/ckpts \
--logdir /data/logs \
--lr_scheduler cosine --project_name sampleproject --batch_size 6 \
--sample_steps 200 --lr 3e-6 --max_epochs 250
```

## Sample training command (1.x)

```
python3 train.py  --ckpt_every_n_minutes 30 --useadam8bit \
--resume_ckpt /data/ckpt_cache/v1-5-pruned-emaonly \
--data_root /data/training_data/family \
--save_ckpt_dir /data/ckpts \
--logdir /data/logs \
--lr_scheduler cosine --project_name family_v1-5 --batch_size 4 \
--sample_steps 200 --lr 3e-6 --max_epochs 25
```

# temp LR try
```
python3 train.py --ckpt_every_n_minutes 30 --resume_ckpt /data/ckpt_cache/v2-1_768-nonema-pruned --data_root /data/training_data/768/ --save_ckpt_dir /data/ckpts --logdir /data/logs --lr_scheduler cosine --project_name tatum_768_lr8e-7 --batch_size 2 --sample_steps 200 --lr 8e-7 --max_epochs 1000 --useadam8bit --resolution 768
```