#
# Trainer and Tools and Laion (Warning, big!)
# 
FROM entmike/everydream:trainer

# Add EveryDream tools
ADD repo/EveryDream /tools
WORKDIR /tools

# Pre-fill tools .cache
ADD .toolcache /tools/.cache

# Install pre-requisites (requirements.txt not needed b/c conda env)
RUN bash -c "git clone https://github.com/salesforce/BLIP scripts/BLIP"
RUN bash -c "source activate everydream && pip install aiofiles timm fairscale colorama"

# Install LAION2B-en Dataset parquets
RUN rm laion/* && \
    git lfs install && \
    git clone https://huggingface.co/datasets/laion/laion2B-en-aesthetic laion

# Finish up
WORKDIR /everydream

# Add welcome banner
ADD welcome-banner.txt /root/welcome-banner.txt
RUN echo "cat ~/welcome-banner.txt" >> ~/.bashrc