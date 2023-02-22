#!/bin/bash
echo "Worker Started"

API_URL=${1:-${API_URL}} 
API_AGENT=${2:-${API_AGENT}}
API_OWNER=${3:-398901736649261056}

echo 🌎 $API_URL
echo 💻 $API_AGENT
echo 😀 $API_OWNER

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

API_URL=$API_URL API_AGENT=$API_AGENT API_OWNER=$API_OWNER python workermode.py
