#!/usr/bin/env bash
set -eu

[[ "$(lsb_release -sc)" == "bionic" ]] || exit 1
ROS_DISTRO=melodic

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

sudo apt install -y curl
curl -k https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -
sudo apt update || echo ""

sudo apt install -y ros-${ROS_DISTRO}-desktop-full python-rosdep
python --version 2>&1 | grep -q "2.7" || exit 1

ls /etc/ros/rosdep/sources.list.d/20-default.list > /dev/null 2>&1 && sudo rm /etc/ros/rosdep/sources.list.d/20-default.list
sudo rosdep init 
rosdep update

sudo apt install -y python-rosinstall python-rosinstall-generator python-wstool build-essential python-catkin-tools python3-vcstool

grep -F "source /opt/ros/${ROS_DISTRO}/setup.bash" ~/.bashrc ||
echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

grep -F `catkin locate --shell-verbs` ~/.bashrc ||
echo "source `catkin locate --shell-verbs`" >> ~/.bashrc

grep -F "ROS_IP" ~/.bashrc ||
echo "export ROS_IP=127.0.0.1" >> ~/.bashrc

grep -F "ROS_MASTER_URI" ~/.bashrc ||
echo "export ROS_MASTER_URI=http://\$ROS_IP:11311" >> ~/.bashrc

echo ""
echo "Success installing ROS ${ROS_DISTRO}"
echo "Run 'source ~/.bashrc'"
echo ""
echo "If any error occurs, please refer to the following URL."
echo "https://github.com/Tiryoh/ros_setup_scripts_ubuntu/"
echo ""
