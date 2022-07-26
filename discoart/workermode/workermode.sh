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

# Download most recent FD model list
cd /
wget https://www.feverdreams.app/models.yaml

# Update and start worker
cd /workermode
pip install -r requirements.txt
pip install discoart --upgrade
wget -O https://raw.githubusercontent.com/entmike/docker-images/main/discoart/workermode/workermode.py

# Use FD model list
export DISCOART_MODELS_YAML='/models.yaml'
python workermode.py --dd_api $1 --agent $2