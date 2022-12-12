FROM python:3.10-slim-bullseye
LABEL maintainer="@EndoNrak"

RUN apt update && \
    apt upgrade -y && \
    apt install -y \
        wget \
        curl \
        git \
        build-essential \
        sudo
# dependencies for GUI 
RUN apt install -y \ 
        libglib2.0-dev \
        libgl1-mesa-dev \
        '^libxcb.*-dev' \
        libx11-xcb-dev \
        libglu1-mesa-dev \
        libxrender-dev \
        libxi-dev \
        libxkbcommon-dev \
        libxkbcommon-x11-dev \
        libdbus-1-dev
 
RUN mkdir /tetris
WORKDIR /tetris
ADD ../requirements.txt .
RUN pip install -r requirements.txt 
