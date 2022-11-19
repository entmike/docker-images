## Example command

```bash
docker run --gpus "device=1" --rm -it \
-p 7860:7860 -p 7861:7861 \
-v /mnt/user/ml/stablediffusion/models/:/home/stable/stable-diffusion-webui/models \
-v /mnt/user/ml/stablediffusion/embeddings/:/home/stable/stable-diffusion-webui/embeddings \
-v /mnt/user/ml/stablediffusion/repositories/:/home/stable/stable-diffusion-webui/repositories \
-v /mnt/user/ml/stablediffusion/outputs:/home/stable/stable-diffusion-webui/outputs \
-v /mnt/user/ml/stablediffusion/training_data:/training_data \
-v /mnt/user/ml/stablediffusion/textual_inversion:/home/stable/stable-diffusion-webui/textual_inversion \
-v /mnt/user/ml/stablediffusion/cache:/home/stable/.cache \
entmike/a1111 bash
```