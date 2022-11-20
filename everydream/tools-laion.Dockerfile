FROM entmike/everydream:tools

WORKDIR /tools
# Install LAION2B-en Dataset parquets
RUN rm laion/* && \
    git lfs install && \
    git clone https://huggingface.co/datasets/laion/laion2B-en-aesthetic laion

WORKDIR /everydream