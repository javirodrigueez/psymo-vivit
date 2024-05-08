FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Base installation
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
  software-properties-common \
  wget \
  ffmpeg \
  libsm6 \
  libxext6 \
  joe

# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=${CUDA_HOME}/bin:${PATH}
# ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
WORKDIR /workspace

# Install python/pip 3.12
RUN add-apt-repository ppa:deadsnakes/ppa && \
  apt-get update && apt-get install -y python3.12 python3.12-distutils python3.12-dev && \
  wget https://bootstrap.pypa.io/get-pip.py && \
  python3.12 get-pip.py && \
  ln -s /usr/bin/python3.12 /usr/local/bin/python && \
  rm -rf /var/lib/apt/lists/* && \
  pip install --upgrade pip urllib3 setuptools six requests

# Install SEVILA dependencies
RUN pip install \
  docopt \
  python-dotenv \
  pandas \
  torch \
  torchvision \
  transformers \
  scikit-learn
