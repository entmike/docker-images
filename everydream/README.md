# EveryDream Docker Container

This folder contains Dockerfiles for EveryDream.  There are different Dockerfiles for specific use cases and any size/bandwidth constraints:

## Trainer
Contains ready-to-run trainer environment.  (Visit https://github.com/victorchall/EveryDream-trainer for documentation.)  See `trainer.Dockerfile` for source code to build yourself or simply `docker pull entmike/everydream:trainer` to use right away.

### Data directory example
It is recommended (but not required if you know what you are doing) to mount all your working data to a single volume mount point such as to container path `/data`.  An example directory structure of your host data directory could be as follows:
```
/data
  /configs
    project1.yaml
    project2.yaml
  /models
    sd_v1-5_vae.ckpt
  /training_samples
    /project1
    /project2
  /log
```

### Docker launch example:
Below is a simple example to get started:
```sh
docker run --rm -it --gpus "device=0" --shm-size=512m \
-v /path/to/your/data:/data \
entmike/everydream:trainer bash
```
*NOTE: In my experience, I had to set `--shm-size=512m` to avoid shared memory errors.*

### Training in Docker example:
Once you are running inside your container, you can begin training similar to the example below:
```sh
python main.py -t -n Project1 \
--logdir /data/log \
--base /data/configs/project1.yaml \
--actual_resume /data/models/sd_v1-5_vae.ckpt \
--data_root /data/training_samples/project1
```

## Tools
Tools come in 2 different varieties.  **Most people should use Basic unless you are using a service like RunPod or similar.**

- **Basic (Recommended self-hosted approach):**

  Contains EveryDream tools to run data engineering tasks for training.  (Visit https://github.com/victorchall/everydream for documentation.)  See `tools.Dockerfile` for source code to build yourself or simply `docker pull entmike/everydream:tools` to use right away.

  Docker launch example:

  ```sh
  docker run --gpus all --rm -it \
  -v /path/to/your/laion/files:/tools/laion \
  -v /path/to/your/data:/data \
  entmike/everydream:tools bash
  ```

  If you are going to be downloading Laion images and are using this image, it is recommended on your host system to predownload the Laion2B parquet files from(https://huggingface.co/datasets/laion/laion2B-en-aesthetic) and volume mount it to container path `/tools/laion` in your Docker run command.

- **With LAION parquets (Only recommended for RunPod or similar):**

  Contains everything in `tools.Dockerfile` but with the addition of predownloading the LAION2B parquet files.  See `tools-laion.Dockerfile` for source code to build yourself or simply `docker pull entmike/everydream:tools-laion` to use right away.

  ⚠️ Warning, this will build a large image and practically should not be the preferred use case for training in a self-hosted environment.  This type of image is more useful for GPU hosting environments with volume mounting constraints such as RunPod to save time (and your money) from downloading the Laion2B parquet files on first run.

  Docker launch example:

  ```sh
  docker run --gpus all --rm -it \
  -v /path/to/your/data:/data \
  entmike/everydream:tools bash
  ```

# TODO: