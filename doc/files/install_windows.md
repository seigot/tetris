# 実行環境(windowsの場合)
<br>
docker for windowsを使います。<br>
<br>
[DockerDeskop](https://docs.docker.com/docker-for-windows/install/)をインストール<br>
パワーシェルを起動<br>
パワーシェル上で以下を実行する<br>

```
docker run -p 6080:80 --shm-size=512m seigott/tetris_game_docker
```
