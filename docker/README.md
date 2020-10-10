# Tetris Game docker

## build

```
docker build -t seigott/tetris_game_docker .
```

## run

```
sudo docker run -p 6080:80 --shm-size=512m seigott/tetris_game_docker
```
