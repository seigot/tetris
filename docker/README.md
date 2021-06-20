# Tetris Game docker

## 実行方法

### step0. dockerをインストールする

[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu) <br>
[Install Docker Desktop on Mac](https://docs.docker.com/docker-for-mac/install/) <br>
[Install Docker Desktop on Windows](https://docs.docker.com/docker-for-windows/install/) <br>

### step1. dockerコンテナを起動する

以下を実行する。

```
sudo docker run -p 6080:80 --shm-size=512m seigott/tetris_game_docker
```

もしリモートログインしながらdockerコンテナ起動し続けたい場合、上記の代わりに以下を実行する。<br>
（terminalからバックグラウンド実行）<br>

```
sudo nohup docker run -p 6080:80 --shm-size=512m seigott/tetris_game_docker &
```

もし`pytorch(v1.4)`インストール済dockerコンテナを使いたい場合、上記の代わりに以下を実行する。<br>

```
sudo docker run -p 6080:80 --shm-size=512m seigott/tetris_game_docker:pytorchv1.4
```

### step2. ブラウザからdockerコンテナにアクセスする

```
localhost:6080
```

リモート環境でdockerコンテナ起動している場合は、上記の代わりに以下を実行する。<br>

```
${IP_ADDRESS}:6080
```

アクセスできたら以下により動作検証する

左下アイコン --> system tools --> terminator

Terminalを立ち上げて以下を実行

```
cd ~/tetris_game
bash start.sh
```

## update docker container

以下を実行する。

```
sudo docker pull seigott/tetris_game_docker
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

