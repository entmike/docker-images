export DOCKER_BUILDKIT=0
docker build -t entmike/discoart:runpod --build-arg base_image=entmike/discoart:preload -f Dockerfile .