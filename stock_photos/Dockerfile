FROM entmike/stable-diffusion:1.4
ADD requirements.txt /workspace/requirements.txt
RUN cd /workspace && pip install -r requirements.txt
ADD go.py /workspace/go.py