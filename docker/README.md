# Tetris Game docker

## 実行方法

### step0. dockerをインストールする

[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu) <br>
[Install Docker Desktop on Mac](https://docs.docker.com/docker-for-mac/install/) <br>
[Install Docker Desktop on Windows](https://docs.docker.com/docker-for-windows/install/) <br>

### step1. dockerコンテナを起動する

tetrisディレクトリで以下を実行する。  
```
docker compose up
```

`pytorch`インストール済コンテナを使いたい場合、上記の代わりに以下を実行する。<br>

```
docker compose -f docker-compose.pytorch.yaml up
```

### step2. dockerコンテナにアクセスする

コンテナ内に入ってターミナルを起動
```
docker exec -it tetris-container bash
```

コンテナ内にアクセスできたら以下により動作検証する

`/tetris`にいることを確認し、以下を実行

```
python start.py
```

### コンテナの停止

ホストOS側のtetrisディレクトリから以下を実行する。

```
docker compose stop
```
この場合には、コンテナは削除されずに残る。

### コンテナの停止、削除
ホストOS側のtetrisディレクトリから以下を実行する。

```
docker compose dwon
```
この場合には、コンテナは削除される。