# FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

# Install requirements
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash

RUN apt-get update && apt-get upgrade -y && apt install software-properties-common -y && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y wget git python3.10-dev python3.10-venv python3-pip zip unzip sudo libsm6 libxext6 ffmpeg git git-lfs supervisor && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 0

# Create a stable diffusion user
RUN useradd -ms /bin/bash stable -p stable && adduser stable sudo
RUN echo "stable ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers.d/stable

USER stable

# Install AUTOMATIC1111/stable-diffusion-webui
RUN bash -c "cd ~ && COMMANDLINE_ARGS='--exit --skip-torch-cuda-test' bash <(wget -qO- https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh)"

# Install xformers (shouldn't take too long since ninja is installed)
RUN bash -c "source /home/stable/stable-diffusion-webui/venv/bin/activate && pip install ninja && \
export FORCE_CUDA=1 && export TORCH_CUDA_ARCH_LIST=8.6 && CUDA_VISIBLE_DEVICES=0 && \
pip install git+https://github.com/facebookresearch/xformers.git#egg=xformers"

# Add venv activation to .bashrc
RUN echo "source /home/stable/stable-diffusion-webui/venv/bin/activate" >> ~/.bashrc

# Add Fever Dreams workermode deps
WORKDIR /home/stable/stable-diffusion-webui
ADD workermode/requirements.txt /home/stable/stable-diffusion-webui/workermode-requirements.txt
RUN bash -c "source /home/stable/stable-diffusion-webui/venv/bin/activate && \
pip install --force-reinstall httpcore==0.15 && \
pip install -r workermode-requirements.txt"

# Add controlnet extension
RUN git clone https://github.com/Mikubill/sd-webui-controlnet /home/stable/stable-diffusion-webui/extensions/sd-webui-controlnet

# Add workermode files
ADD start.sh /start.sh
ADD workermode/workermode.py /home/stable/stable-diffusion-webui
ADD workermode/workermode.sh /home/stable/stable-diffusion-webui
ADD workermode/supervisord.conf /etc/supervisor/conf.d/

# Add welcome banner
ADD welcome-banner.txt /home/stable/welcome-banner.txt
RUN echo "cat ~/welcome-banner.txt" >> ~/.bashrc
RUN mkdir /home/stable/logs

CMD ["/usr/bin/supervisord", "-n"]