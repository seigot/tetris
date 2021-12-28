#!/bin/bash

apt update -y;
apt install -y build-essential libbz2-dev libdb-dev libreadline-dev libffi-dev libgdbm-dev liblzma-dev libncursesw5-dev libsqlite3-dev libssl-dev zlib1g-dev uuid-dev tk-dev

WORK_DIRECTORY="/tmp/python3"
function update_python3(){

    mkdir -p ${WORK_DIRECTORY}
    cd  ${WORK_DIRECTORY}

    wget https://www.python.org/ftp/python/3.9.9/Python-3.9.9.tar.xz
    tar xJf Python-3.9.9.tar.xz
    cd Python-3.9.9
    ./configure
    make
    sudo make install
}

update_python3

