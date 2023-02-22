## Example command

```bash
docker run --gpus "device=1" --rm --name entmike-3090-dev -it \
-e API_URL=http://192.168.1.12:48080 -e API_AGENT=entmike-3090-dev \
-p 7860:7860 -p 7861:7861 \
-v /mnt/user/work/models/:/home/stable/stable-diffusion-webui/models \
-v /mnt/user/ml/feverdreams/outdir2/:/outdir \
-v /mnt/user/work/automatic1111/embeddings/:/home/stable/stable-diffusion-webui/embeddings \
-v /mnt/user/work/automatic1111/repositories/:/home/stable/stable-diffusion-webui/repositories \
-v /mnt/user/work/automatic1111/outputs:/home/stable/stable-diffusion-webui/outputs \
-v /mnt/user/work/automatic1111/training_data:/training_data \
-v /mnt/user/work/automatic1111/textual_inversion:/home/stable/stable-diffusion-webui/textual_inversion \
-v /mnt/user/work/automatic1111/.cache:/home/stable/.cache \
-v /mnt/user/work/automatic1111/controlnet-models/:/home/stable/stable-diffusion-webui/extensions/sd-webui-controlnet/models \
entmike/sd-a1111-controlnet
```