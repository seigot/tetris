# Tetris Game docker

## run

### step0. dockerをインストールする

[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu)

### step1. dockerコンテナを起動する

```
sudo docker run -p 6080:80 --shm-size=512m seigott/tetris_game_docker
```

### step2. ブラウザからdockerコンテナにアクセスする

```
http://127.0.0.1:6081
```

アクセスできたら以下により動作検証する

左下アイコン --> system tools --> terminator

Terminalを立ち上げて以下を実行

```
cd ~/tetris_game
python3 tetris_game.py
```

## build

```
docker build -t seigott/tetris_game_docker .
```
