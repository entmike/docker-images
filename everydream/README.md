# EveryDream Docker Container

This folder contains Dockerfiles for EveryDream.  There are different Dockerfiles for specific use cases and any size/bandwidth constraints:

## Trainer
Contains ready-to-run trainer environment.  (Visit https://github.com/victorchall/EveryDream-trainer for documentation.)  See `trainer.Dockerfile` for source code to build yourself or simply `docker pull entmike/everydream:trainer` to use right away.

Docker launch example:
```
docker run --rm -it --gpus "device=0" --shm-size=512m \
-v /path/to/your/data:/data \
entmike/everydream:trainer bash
```

## Tools
Tools come in 2 different varieties:

- **Basic (Recommended self-hosted approach):**

  Contains EveryDream tools to run data engineering tasks for training.  (Visit https://github.com/victorchall/everydream for documentation.)  See `tools.Dockerfile` for source code to build yourself or simply `docker pull entmike/everydream:tools` to use right away.

  Docker launch example:

  ```sh
  docker run --gpus all --rm -it \
  -v /path/to/your/laion/files:/tools/laion \
  -v /path/to/your/data:/data \
  entmike/everydream:tools bash
  ```

  If you are going to be downloading Laion images and are using this image, it is recommended to download the Laion2B parquet files (https://huggingface.co/datasets/laion/laion2B-en-aesthetic) and volume mount it to container path `/tools/laion` in your Docker run command.

- **With LAION parquets (Recommended for RunPod):**

  Contains everything in `tools.Dockerfile` but with the addition of predownloading the LAION2B parquet files.  See `tools-laion.Dockerfile` for source code to build yourself or simply `docker pull entmike/everydream:tools-laion` to use right away.

  ⚠️ Warning, this will build a large image and practically should not be the preferred use case for training in a self-hosted environment.  This type of image is more useful for GPU hosting environments with volume mounting constraints such as RunPod to save time (and your money) from downloading the Laion2B parquet files on first run.




## Training
```sh
python main.py -t -n girls \
--logdir /data/log \
--base /data/training_data/processed/config.yaml \
--actual_resume /data/models/sd_v1-5_vae.ckpt \
--data_root /data/training_data/processed
```