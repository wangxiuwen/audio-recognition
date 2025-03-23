# 基于 Nvidia CUDA 11.8 镜像
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/opt/conda/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 更换 Ubuntu 源为清华镜像，加速 apt-get
RUN echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse' > /etc/apt/sources.list \
    && echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse' >> /etc/apt/sources.list \
    && echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse' >> /etc/apt/sources.list \
    && echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse' >> /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y wget curl git bzip2 libasound2 vim \
    && rm -rf /var/lib/apt/lists/*


# 安装 Miniconda 并创建 Python 3.13 环境
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda config --set show_channel_urls yes \
    && conda config --add channels conda-forge \
    && conda update -n base -c defaults conda \
    && conda create -y -n audio python=3.13 \
    && conda clean -afy

RUN wget --quiet https://hf-mirror.com/csukuangfj/sherpa-onnx-wheels/resolve/main/cuda/1.11.2/sherpa_onnx-1.11.2+cuda-cp313-cp313-linux_x86_64.whl  -O /tmp/sherpa_onnx-1.11.2+cuda-cp313-cp313-linux_x86_64.whl \
    && conda run -n audio pip install /tmp/sherpa_onnx-1.11.2+cuda-cp313-cp313-linux_x86_64.whl

# 激活 Conda 环境
SHELL ["/bin/bash", "-c"]
RUN conda init bash \
    && echo "conda activate audio" >> ~/.bashrc

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . .

# 安装 Python 依赖
RUN conda run -n audio pip install --upgrade pip setuptools wheel \
    && conda run -n audio pip install fastapi \
    && conda run -n audio pip install uvicorn \
    && conda run -n audio pip install gunicorn \
    && conda run -n audio pip install soundfile \
    && conda run -n audio pip install coloredlogs \
    && conda run -n audio pip install jinja2 \
    && conda run -n audio pip install python-multipart \
    && conda run -n audio pip install torch \
    && conda run -n audio pip install librosa \
    && conda run -n audio pip install onnx \
    && conda run -n audio pip install onnxruntime-gpu \
    && rm -rf ~/.cache/pip


# 暴露端口
EXPOSE 29999

CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate audio && exec gunicorn --config gunicorn_conf.py main:app"]
