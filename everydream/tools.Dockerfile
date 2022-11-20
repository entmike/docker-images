#
# Tools Image
#
FROM entmike/everydream:base

# Install Python (maybe swap to conda later)
RUN apt-get update && apt-get install -y \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add EveryDream tools
ADD repo/EveryDream /tools
WORKDIR /tools

# Pre-fill tools .cache
ADD .toolcache /tools/.cache

# Install pre-requisites
RUN bash -c "git clone https://github.com/salesforce/BLIP scripts/BLIP"
RUN bash -c "pip install -r requirements.txt"

# Finish up
WORKDIR /tools

# Add welcome banner
ADD welcome-tools-banner.txt /root/welcome-tools-banner.txt
RUN echo "cat ~/welcome-tools-banner.txt" >> ~/.bashrc