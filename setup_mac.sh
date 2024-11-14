#!/bin/bash

# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install necessary packages
brew install python pyqt5 numpy
pip3 install -U pip
pip3 install torch tensorboardX

# Clone the repository and run initial setup
git clone https://github.com/seigot/tetris
cd tetris
python3 start.py -l1
