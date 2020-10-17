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
localhost:6080
```

アクセスできたら以下により動作検証する

左下アイコン --> system tools --> terminator

Terminalを立ち上げて以下を実行

```
cd ~/tetris_game
bash start.sh
```

## build for debug

```
docker build -t seigott/tetris_game_docker .
```
