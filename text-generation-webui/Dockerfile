FROM huggingface/transformers-pytorch-gpu

# Install base utilities
RUN apt-get update && apt-get install -y build-essential wget
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

VOLUME /data

RUN git clone https://github.com/oobabooga/text-generation-webui /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN rm -rf /app/models
RUN ln -s /data /app/models

ENV PATH=$PATH:/app