# 実行環境(windowsの場合)
docker for windowsを使います。<br>

### step0. DockerDesktopのインストール 

[DockerDeskop](https://docs.docker.com/docker-for-windows/install) をインストール<br>

### step1. パワーシェル上でコンテナを起動する<br>

パワーシェルを起動して以下実行<br>

```
docker run -p 6080:80 --shm-size=512m seigott/tetris_game_docker
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
