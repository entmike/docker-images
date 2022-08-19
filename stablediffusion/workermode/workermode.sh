#!/bin/bash
echo "Worker Started"

if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo $PUBLIC_KEY >> authorized_keys
    chmod 700 -R ~/.ssh
    cd /
    service ssh start
    echo "SSH Service Started"
fi

# Update and start worker
cd /workspace
curl -O https://raw.githubusercontent.com/entmike/docker-images/main/stablediffusion/workermode/workermode.py

python workermode.py --dd_api $1 --agent $2 --owner ${3:-398901736649261056} --ckpt ${4:/weights/sd-v1-3-full-ema.ckpt} --config ${5:/workspace/k-diffusion/v1-inference.yaml}
