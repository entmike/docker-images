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
cd /workermode
pip install -r requirements.txt
curl -O https://raw.githubusercontent.com/entmike/docker-images/main/discoart/workermode/workermode.py

# Use FD model list
# export DISCOART_MODELS_YAML='/models.yaml' # Nope.
# export DISCOART_DISABLE_REMOTE_MODELS='1' # disable the listing of diffusion models on Github, remote diffusion models allows user to use latest models without updating the codebase.  # Nope
export DISCOART_REMOTE_MODELS_URL='https://www.feverdreams.app/models.yaml' # use a custom remote URL for fetching models list
python workermode.py --dd_api $1 --agent $2 --owner ${3:-398901736649261056}
