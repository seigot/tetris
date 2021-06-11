#!/bin/bash

HOME=/home/ubuntu
source $HOME/.bashrc

function install_package(){
    echo "install packages"

    sudo apt-get install -y apt-utils
    sudo apt-get update
}
install_package

#function install_ai_package(){
#    echo "install packages"
#    source /opt/ros/melodic/setup.bash
#
#    sudo apt install -y libignition_math
#    # gazebo/gzserver setting
#    echo "export QT_X11_NO_MITSHM=1" >> $HOME/.bashrc
#    sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
#
#    # setup repository
#    ## git clone
#    pushd ~/catkin_ws/src/
#    git clone http://github.com/kenjirotorii/ai_race
#    cd ai_race
#    git checkout 89a21f0a59961000ef231b7a04c055cf9e2fafbf
#    ## add patch
#    git clone http://github.com/seigot/ai_race ~/tmp/ai_race_tmp
#    patch -p1 < ~/tmp/ai_race_tmp/docker/jetson/kenjirotorii_wheel_robot.urdf.xacro.patch
#    rm -r ~/tmp/ai_race_tmp
#    ## build
#    cd ../..
#    catkin build
#    source devel/setup.bash
#    popd
#}
#install_ai_package

# pyqt5
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-pyqt5
pip3 install --upgrade pip
pip3 install numpy
cd $HOME
git clone http://github.com/seigot/tetris_game


#コンテナを起動し続ける
#tail -f /dev/null
