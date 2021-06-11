#!/bin/bash -x

# cuda10.1, cudnn7のdocker環境でshが通ることを確認
# train.pyの動作も確認済み
# Docker imageはnvidia/cuda:10.1-cudnn7-devel-ubuntu18.04を使用
# 参考 https://qiita.com/nakasuke_/items/ec1b0636416df3c72db3#docker%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB

# エラーが起こったら異常終了させる
set -E

function failure(){
    echo "error end!!"
    exit 1
}

# エラー発生時にコールする関数を設定 
trap failure ERR

#DEBIAN_FRONTEND=noninteractive

function setup_swap_file(){
    cd ~
    if [ -d "./installSwapfile" ]; then
	echo "skip"
	return 0
    fi
    git clone https://github.com/JetsonHacksNano/installSwapfile
    cd installSwapfile
    ./installSwapfile.sh
    # SWAP領域が増えていることを確認
    free -mh
}

function install_basic_package(){

    # prepare
    wget  http://packages.osrfoundation.org/gazebo.key
    sudo apt-key add gazebo.key

    sudo apt-get update
    sudo apt-get install -y net-tools git
    sudo apt-get install -y python-pip
    # install pyqt5 and NumPy
    sudo apt-get install -y python3-pip
    sudo apt-get install -y python3-pyqt5
    pip install future
    pip3 install future
    pip3 install --upgrade pip
    pip install numpy
    pip3 install numpy
    # for judge server
    pip3 install flask
    pip3 install requests
    python -m pip install requests
    # pygame
    sudo apt-get update -y
    sudo apt-get install -y libsdl-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev
    sudo apt-get install -y libsmpeg-dev libportmidi-dev libavformat-dev libswscale-dev
    sudo apt-get install -y libfreetype6-dev
    sudo apt-get install -y libportmidi-dev
    sudo pip3 install pgzero
    #python -m pip install pygame==1.9.6
    # scikit learn
    sudo apt install -y gfortran
}

function install_ros(){
    # check if already install ros
    #if [ ! -z `rosversion -d` ];then
    #	echo "ros already installed, skip install ros"
    #	return 0
    #fi

    cd ~
    # sudo rm -rf jetson-nano-tools
    # git clone https://github.com/karaage0703/jetson-nano-tools
    # cd jetson-nano-tools
    # ./install-ros-melodic.sh
    # [future work memo] install lsb-release, then use..
    # apt-get update -y;apt-get install -y lsb-release;
    # sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    ROS_DISTRO=melodic
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list'
    sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
    sudo apt update
    sudo apt install -y ros-melodic-desktop-full
    sudo apt-get install -y python-catkin-tools
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws
    source /opt/ros/melodic/setup.bash
    catkin init
    catkin build
    sh -c "echo \"source ~/catkin_ws/devel/setup.bash\" >> ~/.bashrc"
    source ~/.bashrc
    sudo apt-get install -y python-rosdep python-rosinstall python-rosinstall-generator build-essential

    # 暫定でコメントアウト
    # Dockerfileの方で実行しているため。本来はここでrosdep initしたい。
    #sudo rosdep init
    #rosdep update

    echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
    source /opt/ros/melodic/setup.bash
}

function install_ros_related_packages(){
    # joint state controller, and ros package
    sudo apt install -y ros-melodic-ros-control ros-melodic-ros-controllers  ros-melodic-joint-state-controller ros-melodic-effort-controllers ros-melodic-position-controllers ros-melodic-joint-trajectory-controller
    sudo apt install ros-melodic-cob-srvs
    # gazebo
    sudo apt-get install -y gazebo9
    sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
    wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
    sudo apt-get update -y
    sudo apt-get install -y ros-melodic-gazebo-ros-pkgs ros-melodic-gazebo-ros-control
    echo "export GAZEBO_MODEL_PATH=:${HOME}/catkin_ws/src/ai_race/ai_race:${HOME}/catkin_ws/src/ai_race/ai_race/sim_world/models" >> ~/.bashrc
    export GAZEBO_MODEL_PATH=:${HOME}/catkin_ws/src/ai_race/ai_race:${HOME}/catkin_ws/src/ai_race/ai_race/sim_world/models
    # camera image
    sudo apt-get install -y ros-melodic-uvc-camera
    sudo apt-get install -y ros-melodic-image-*
}

function install_torch(){
    ### pytorch from pip image (v1.4)
    sudo apt-get install -y libopenblas-base libopenmpi-dev
    sudo apt-get -y install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
    
    #python -m pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp27-cp27mu-linux_x86_64.whl
    python -m pip install torchvision==0.2.2
    pip3 install torch==1.4.0 torchvision==0.2.2
    pip install 'pillow<7'
}

function install_torch2trt(){
    ### torch2trt
    cd ~
    sudo rm -rf torch2trt
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt
    git checkout d1fa6f9f20c6c4c57a9486680ab38c45d0d94ec3
    sudo python setup.py install
    sudo python3 setup.py install
}

function install_sklearn(){
    ### sklearn python3
    pip3 install scikit-learn
    # pip3 install matplotlib
    #sudo apt-get -y install python3-tk
}

function install_numpy(){
    echo "skip"
    ### pandas python2,3 (defaultを使えばよい)
    #pip3 install cython
    #pip3 install numpy
    pip3 install -U pandas
}

function install_opencv(){
    pip3 install opencv-python==3.4.10.37
    ### opencv python
    ### opencv python はソースからビルドする必要がある. 8～10時間ほど掛かる.
    # cd ~
    # sudo rm -rf nano_build_opencv
    # git clone https://github.com/mdegans/nano_build_opencv
    # cd nano_build_opencv
    # yes | ./build_opencv.sh 3.4.10
}

function setup_this_repository(){

    echo "install packages"
    source /opt/ros/melodic/setup.bash

    sudo apt upgrade -y libignition-math2
    # gazebo/gzserver setting
    echo "export QT_X11_NO_MITSHM=1" >> $HOME/.bashrc
    sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'

    # setup repository
    ## git clone
    pushd ~/catkin_ws/src/
    git clone http://github.com/kenjirotorii/ai_race
    cd ai_race
    git checkout 89a21f0a59961000ef231b7a04c055cf9e2fafbf
    ## add patch
    git clone http://github.com/seigot/ai_race ~/tmp/ai_race_tmp
    patch -p1 < ~/tmp/ai_race_tmp/docker/jetson/kenjirotorii_wheel_robot.urdf.xacro.patch
    rm -r ~/tmp/ai_race_tmp
    ## build
    cd ../..
    catkin build
    source devel/setup.bash
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
    popd

#    mkdir -p ~/Images_from_rosbag
#    cd ~/catkin_ws/src
#
#    if [ -d "./ai_race" ]; then
#	echo "skip ai_race directory already exist.."
#	return 0
#    fi
#    echo "clone sample repository.."
#    git clone http://github.com/seigot/ai_race
#    cd ~/catkin_ws
#    catkin build
#    source devel/setup.bash
#    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
}

function check_lib_version(){
    python3 -c 'import torch; print(torch.__version__) '
    python3 -c "import torchvision;print(torchvision.__version__);"
    python3 -c "import cv2 ;print(cv2.__version__);"
    python3 -c "import sklearn;print(sklearn.__version__);"
    python3 -c "import pandas as pd ;print(pd.__version__);"
    python -c 'import torch; print(torch.__version__) '
    python -c "import torchvision;print(torchvision.__version__);"
    python -c "import cv2 ;print(cv2.__version__);"
}

echo "start install"
# setup_swap_file
install_basic_package
install_ros
install_ros_related_packages
install_torch
# install_torchvision
# install_torch2trt
install_sklearn
install_numpy
install_opencv
setup_this_repository
check_lib_version
echo "finish install"
