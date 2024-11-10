# syntax=docker/dockerfile:1

# =============================================================
# Configuration
# =============================================================

ARG UBUNTU_VERSION=20.04
ARG CUDA_MAJOR_VERSION=11.8.0
ARG CUDNN_MAJOR_VERSION=8
ARG PYTHON_VERSION=3.10.14
ARG PYTORCH_VERSION=2.1.2
ARG TORCHVISION_VERSION=0.16.2
ARG NUMPY_VERSION=1.26.4
ARG BUILD_JOBS=16

# =============================================================
# Create docker
# =============================================================

FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

# propagate build args
ARG CUDA_MAJOR_VERSION
ARG PYTHON_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG NUMPY_VERSION
ARG BUILD_JOBS

# configure environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

# configure timezone
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install libs
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        git \
        vim \
        screen \
        curl \
        wget \
        xz-utils \
        build-essential \
        libgomp1 \
        libjpeg-turbo8 \
        libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev gcovr libffi-dev uuid-dev \
        libgtk2.0-dev libgsf-1-dev libtiff5-dev libopenslide-dev \
        libgl1-mesa-glx libgirepository1.0-dev libexif-dev librsvg2-dev fftw3-dev orc-0.4-dev

# install python with up-to-date pip
RUN cd /tmp && \
    wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" && \
    tar xfv Python*.xz && \
    cd Python-3*/ && \
    ./configure --enable-shared LDFLAGS="-fprofile-arcs" && \
    make -j${BUILD_JOBS} install && \
    cd ~ && \
    rm -rf /tmp/Python-3* && \
    ldconfig

RUN --mount=type=cache,mode=0777,target=/home/.cache/pip python3 -m pip install --upgrade pip==24.0 pip-tools wheel setuptools && \
    printf '#!/bin/bash\necho "Please use pip3 instead of pip to install packages for python3"' > /usr/local/bin/pip && \
    chmod +x /usr/local/bin/pip 

# install ASAP
# RUN apt-get update && \
#     apt-get -y install curl git && \
#     curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1-(Nightly)/ASAP-2.1-Ubuntu2004.deb" && \
#     dpkg --install ASAP-2.1-Ubuntu2004.deb || true && \
#     apt-get -f install --fix-missing --fix-broken --assume-yes && \
#     ldconfig -v && \
#     apt-get clean && \
#     echo "/opt/ASAP/bin" > /usr/local/lib/python3.9/site-packages/asap.pth && \
#     rm ASAP-2.1-Ubuntu2004.deb

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# install python libraries
WORKDIR /opt/app

# create output directory
RUN mkdir /opt/app/data
RUN chown user:user /opt/app/data

COPY --chown=user:user requirements.in /opt/app/

COPY --chown=user:user configs /opt/app/configs
COPY --chown=user:user dataset_csv /opt/app/dataset_csv
COPY --chown=user:user datasets /opt/app/datasets
COPY --chown=user:user mamba /opt/app/mamba
COPY --chown=user:user models /opt/app/models
COPY --chown=user:user MyCallback /opt/app/MyCallback
COPY --chown=user:user MyLoss /opt/app/MyLoss
COPY --chown=user:user MyOptimizer /opt/app/MyOptimizer
COPY --chown=user:user splits /opt/app/splits
COPY --chown=user:user utils /opt/app/utils

RUN --mount=type=cache,mode=0777,target=/root/.cache/pip cd /opt/app/ && \
    CUDA_IDENTIFIER_PYTORCH=`echo "cu${CUDA_MAJOR_VERSION}" | sed "s|\.||g" | cut -c1-5` && \
    sed -i \
        -e "s|%NUMPY_VERSION%|${NUMPY_VERSION}|g" \
        -e "s|%PYTORCH_VERSION%|${PYTORCH_VERSION}+${CUDA_IDENTIFIER_PYTORCH}|g" \
        -e "s|%TORCHVISION_VERSION%|${TORCHVISION_VERSION}+${CUDA_IDENTIFIER_PYTORCH}|g" \
        requirements.in && \
    /usr/local/bin/python3 -m piptools compile requirements.in --verbose --find-links https://download.pytorch.org/whl/torch_stable.html && \
    /usr/local/bin/python3 -m piptools sync

RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

RUN cd /opt/app/mamba && pip3 install .
RUN pip3 install causal-conv1d==1.4.0

COPY --chown=user:user train.py /opt/app/
COPY --chown=user:user run.sh /opt/app/

USER user
ENTRYPOINT ["bash", "run.sh"]