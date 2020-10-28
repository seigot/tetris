# Tetris Game docker

## 実行方法

### step0. dockerをインストールする

[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu) <br>
[Install Docker Desktop on Mac](https://docs.docker.com/docker-for-mac/install/) <br>
[Install Docker Desktop on Windows](https://docs.docker.com/docker-for-windows/install/) <br>

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

## [開発用] build for update docker container

[Dockerfile](./Dockerfile)を更新して以下を実行する

```
docker build -t seigott/tetris_game_docker .
```

コンテナ登録は以下

```
docker login
docker push seigott/tetris_game_docker
docker logout
```

