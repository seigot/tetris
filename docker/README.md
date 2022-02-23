# Tetris Game docker

## 実行方法

### step0. dockerをインストールする

[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu) <br>
[Install Docker Desktop on Mac](https://docs.docker.com/docker-for-mac/install/) <br>
[Install Docker Desktop on Windows](https://docs.docker.com/docker-for-windows/install/) <br>

### step1. dockerコンテナを起動する

以下を実行する。

```
sudo docker run -p 6080:80 --shm-size=512m seigott/tetris_docker
```

もしリモートログインしながらdockerコンテナ起動し続けたい場合、上記の代わりに以下を実行する。<br>
（terminalからバックグラウンド実行）<br>

```
sudo nohup docker run -p 6080:80 --shm-size=512m seigott/tetris_docker &
```

もし`pytorch(v1.10)`インストール済dockerコンテナを使いたい場合、上記の代わりに以下を実行する。<br>

```
sudo docker run -p 6080:80 --shm-size=512m seigott/tetris_docker:pytorchv1.10
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
cd ~/tetris
python start.py
```

## update docker container

以下を実行する。

```
sudo docker pull seigott/tetris_docker
```

## [開発用] build for update docker container

[Dockerfile](./Dockerfile)を更新して以下を実行する

```
# 通常版の場合
docker build -t seigott/tetris_docker .

# pytorch版の場合
docker build -f ./Dockerfile.pytorchv1.10 -t seigott/tetris_docker:pytorchv1.10 .
```

コンテナ登録は以下

```
docker login
docker push seigott/tetris_docker
docker push seigott/tetris_docker:pytorchv1.10 .
docker logout
```

python versionは以下

```
# see install_python.sh in detail
python3 --version
3.9.9
```

