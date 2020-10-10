#!/bin/bash

HOME=/home/ubuntu
source $HOME/.bashrc
source /opt/ros/kinetic/setup.bash

function install_package(){
    echo "install packages"

    sudo apt-get install -y apt-utils
    sudo apt-get update
}
install_package

function install_package2(){
    echo "install packages"

    # AI
    #pip install tensorflow==1.5.0 keras==2.2.5
}
install_package2

# gazebo/gzserver setting
echo "export QT_X11_NO_MITSHM=1" >> $HOME/.bashrc
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'

# pyqt5
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-pyqt5
pip3 install --upgrade pip
pip3 install numpy
cd $HOME
git clone http://github.com/seigot/tetris_game


#コンテナを起動し続ける
#tail -f /dev/null
