#!/bin/bash
echo "Worker Started"
echo $1
echo $2
echo $3
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

API_URL=$1 API_AGENT=$2 API_OWNER=${3:-398901736649261056} python workermode.py --ckpt /weights/v1-5-pruned-emaonly.ckpt
