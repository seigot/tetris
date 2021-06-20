#!/bin/bash -x

# エラーが起こったら異常終了させる
set -E

function failure(){
    echo "error end!!"
    exit 1
}

# エラー発生時にコールする関数を設定 
trap failure ERR

function install_emacs(){
    sudo apt install -y emacs
}

function install_nano(){
    sudo apt install -y nano
}

function install_gedit(){
    sudo apt install -y gedit
}

function install_vscode(){
    echo "install vscode"
    sudo apt install curl
    curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
    sudo install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/
    sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
    cat /etc/apt/sources.list.d/vscode.list
    sudo apt install apt-transport-https
    sudo apt update
    sudo apt list -a code
    sudo apt install code
    code --version --user-data-dir /home/ubuntu
}

echo "start install editor"
install_emacs
install_nano
install_gedit
install_vscode
echo "finish install editor"
