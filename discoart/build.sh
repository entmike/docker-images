export DOCKER_BUILDKIT=0
docker build -t entmike/discoart --build-arg base_image=nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04 -f Dockerfile .