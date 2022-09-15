# Feverdreams.app Docker images

The feverdreams.app repository for entmike's Docker images.

![An example of AI-generated art](discoart/examples/1-getting-started/getting-started-0.png)

## Steps to build

1. Clone the repository
2. `cd` into the repository
3. View the README.md file

### DiscoArt Docker Images

These are found in the `docker-images/discoart` directory.

1. Build images from the Dockerfiles in this repository using `./build-all.sh`.

    ```bash
    cd docker-images/discoart
    ./build-all.sh
    ```

    This may take a while depending on your system resources. You can see the progress in the Docker build logs.

2. Check the images have been created using the `docker images` command.

    ```bash
    docker images
    ```

3. Push the images to a Docker registry using `./push.sh` and `./push-extra.sh`.

    ```bash
    ./push.sh
    ./push-extra.sh
    ```

4. If you are building your own Docker images with prebaked models, follow the instructions in the README.md files in: 
   1. `/models/discoart/base`,
   2. `/models/discoart/extra`,
   3. `/models/clip/base`, and
   4. `/models/clip/extra`.

    You will need to download the listed models and place them in those folders.

5. To start the SSH services and the Jupyter Lab, make sure you have both the public key and Jupiter password set as environment variables. Then run:

    ```bash
    ./start.sh
    ```
