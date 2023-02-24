#!/bin/bash
echo "Container Started"

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

CONTROLNET_COMMIT=${CONTROLNET_COMMIT:-b0c6d973380eb8fdd2d53387ebce4071cb1e8e5b}

if [[ $JUPYTER_PASSWORD ]]
then
    cd /
    jupyter lab --allow-root --no-browser --port=8888 --ip=* \
        --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
        --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=/workspace
    echo "Jupyter Lab Started"
else
    # Update A1111
    git pull
    # Update ControlNet
    cd /home/stable/stable-diffusion-webui/extensions/sd-webui-controlnet
    # git pull
    git checkout $CONTROLNET_COMMIT
    # Launch A1111
    cd /home/stable/stable-diffusion-webui
    python launch.py --listen --no-half --xformers --enable-insecure-extension-access \
    --api --no-download-sd-model --api-log --enable-console-prompts
fi
