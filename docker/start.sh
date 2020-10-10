#!/bin/bash

HOME=/home/ubuntu
source $HOME/.bashrc
source /opt/ros/kinetic/setup.bash

function install_package(){
    # turtlebot3
    sudo apt-get install -y apt-utils
    sudo apt-get update

    sudo apt-get install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential python-pip ros-kinetic-turtlebot3 ros-kinetic-turtlebot3-msgs ros-kinetic-turtlebot3-simulations
    pip install requests flask

    # additional package
    sudo apt-get install -y ros-kinetic-dwa-local-planner
    sudo apt-get install -y ros-kinetic-global-planner
    sudo apt install -y libarmadillo-dev libarmadillo6
}
install_package

function install_package2(){
    # bocchi
    #pip install numpy==1.16.6 \
    #    scipy==1.2.1 \
    #    gym==0.16.0 \
    #    Markdown==3.1.1 \
    #    setuptools==44.1.1 \
    #    grpcio==1.27.2 \
    #    mock==3.0.5 \
    #    tensorflow==1.14.0 \
    #    tensorflow-gpu==1.14.0 \
    #    keras==2.3.0 \
    #    flatten_json==0.1.7
    sudo apt-get install -y ros-kinetic-opencv-apps
    
    # maru-mi
    #pip install \
    #	#Keras==2.2.4 \
    #	keras-rl==0.4.2 \
    #	matplotlib==2.2.5 \
    #	opencv-python==4.2.0.32
    #numpy==1.16.6 \
	
    #tensorflow==1.14.0
    
    # AI
    sudo apt install -y ros-kinetic-dwa-local-planner
    pip install tensorflow==1.5.0 keras==2.2.5
    
    # Bola de arroz
    sudo apt install -y ros-kinetic-libg2o
    sudo apt install -y libopencv-dev
    sudo apt install -y ros-kinetic-costmap-converter
    sudo apt install -y libsuitesparse-dev
    sudo apt install -y libarmadillo-dev libarmadillo6
    
    # ikepoyo
    sudo apt-get install -y ros-kinetic-dwa-local-planner
    sudo apt-get install -y libarmadillo-dev libarmadillo6
    
    # gantetsu
    sudo apt-get install -y ros-kinetic-dwa-local-planner
    sudo apt-get install -y ros-kinetic-slam-gmapping
    pip install transitions
    sudo apt-get install -y graphviz graphviz-dev
    pip install graphviz
    sudo apt install -y ros-kinetic-executive-smach
    rospack find smach
    sudo apt-get install --no-install-recommends -y libarmadillo-dev libarmadillo6
    
    # daito
    sudo apt-get install -y ros-kinetic-dwa-local-planner 
    sudo apt-get install -y ros-kinetic-jsk-visualization
    
    # anko
    sudo apt-get install -y ros-kinetic-dwa-local-planner ros-kinetic-geometry2
    
    # indo carry
    #pip install opencv-python==4.2.0.32
    sudo apt-get install -y ros-kinetic-vision-opencv
    sudo apt-get install -y python-opencv
    sudo apt-get install -y libopencv-dev
    sudo apt-get install -y ros-kinetic-cv-camera
}
install_package2

# workspace作成
#mkdir -p $HOME/catkin_ws/src
#cd $HOME/catkin_ws/src
#catkin_init_workspace
#cd $HOME/catkin_ws/
#catkin_make
#echo "source ~/catkin_ws/devel/setup.bash" >> $HOME/.bashrc
#source $HOME/.bashrc

# Turtlebot3のモデル名の指定を環境変数に追加。
echo "export GAZEBO_MODEL_PATH=$HOME/catkin_ws/src/burger_war/burger_war/models/" >> $HOME/.bashrc
echo "export TURTLEBOT3_MODEL=burger" >> $HOME/.bashrc
source $HOME/.bashrc

# make
#cd $HOME/catkin_ws/src
#git clone https://github.com/pal-robotics/aruco_ros
#cd $HOME/catkin_ws
#catkin_make

# onenightrobocon
#cd $HOME/catkin_ws/src
#git clone https://github.com/OneNightROBOCON/burger_war
#mv burger_war burger_war.org
#cd $HOME/catkin_ws
#catkin_make

mkdir -p $HOME/catkin_ws/src
cd $HOME/catkin_ws/src
git clone https://github.com/pal-robotics/aruco_ros   # arco
#git clone https://github.com/OneNightROBOCON/burger_war # onenightrobocon
git clone https://github.com/seigot/burger_war
git clone https://github.com/tysik/obstacle_detector.git # obstacle detector
git clone https://github.com/seigot/burger_war_autotest2
cd $HOME/catkin_ws
catkin build
echo "source ~/catkin_ws/devel/setup.bash" >> $HOME/.bashrc
source $HOME/.bashrc


# gazebo/gzserver setting
echo "export QT_X11_NO_MITSHM=1" >> $HOME/.bashrc
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install gazebo7 -y
source $HOME/.bashrc


# pyqt5
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-pyqt5
pip3 install --upgrade pip
pip3 install numpy
cd $HOME
git clone http://github.com/seigot/tetris_game


#コンテナを起動し続ける
#tail -f /dev/null
