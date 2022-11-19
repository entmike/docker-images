# EveryDream Docker Container

## Launching Docker Container
```
docker run --rm -it \
--gpus "device=0" --shm-size=256m \
-v /mnt/user/public/:/data \
-v $(pwd)/.cache:/root/.cache \
entmike/everydream bash
```

## Training
```sh
conda activate everydream
python main.py -t -n MyProjectName \
--logdir /data/log \
--base configs/stable-diffusion/v1-finetune_everydream.yaml \
--actual_resume /data/sd_v1-5_vae.ckpt \
--data_root /data/training_data/myproject
```