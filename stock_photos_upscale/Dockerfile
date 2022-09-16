FROM entmike/real-esrgan
ADD requirements.txt /workspace/requirements.txt
RUN cd /workspace && pip install -r requirements.txt
ADD go.py /workspace/go.py
ADD watermark.png /workspace/watermark.png