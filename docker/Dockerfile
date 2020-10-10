FROM tiryoh/ros-desktop-vnc:kinetic
ENV DEBIAN_FRONTEND noninteractive

# update
RUN apt-get update ;\
    apt-get -y upgrade ;

# environment setting
#RUN locale-gen en_US.UTF-8  
#ENV LANG en_US.UTF-8  
#ENV LANGUAGE en_US:en  
#ENV LC_ALL en_US.UTF-8
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:jp
ENV LC_ALL ja_JP.UTF-8

# tools
RUN apt-get update ;\
    apt-get install -y \
    sudo \
    lsb-release \
    terminator \
    gnome-terminal \
    emacs24;

# turtlebot3
RUN apt-get update ;\
    apt-get install -y \
    python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential \
    python-pip \
    ros-kinetic-turtlebot3 ros-kinetic-turtlebot3-msgs ros-kinetic-turtlebot3-simulations; \
    pip install requests flask;

# command
ADD  start.sh  /
RUN  chmod +x /start.sh
RUN  /start.sh
