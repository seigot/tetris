# Tetris Game docker

## step0. docker をインストールする

[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu) <br>
[Install Docker Desktop on Mac](https://docs.docker.com/docker-for-mac/install/) <br>
[Install Docker Desktop on Windows](https://docs.docker.com/docker-for-windows/install/) <br>

## step1. docker コンテナを起動する

tetris ディレクトリで以下を実行する。

```
docker-compose up
```

`pytorch`インストール済コンテナを使いたい場合、上記の代わりに以下を実行する。<br>

```
docker-compose -f docker-compose.pytorch.yaml up
```

## step2. docker コンテナにアクセスする

コンテナ内に入ってターミナルを起動

```
docker exec -it tetris-container bash
```

## step3. 起動ホスト OS 側で GUI 起動のための準備

docker コンテナ内でテトリスプログラムを実行する場合、GUI を表示させるためにはホスト OS 側で[x サーバ](https://qiita.com/kakkie/items/c6ccce13ce0beaefaad1)を起動する必要があります。  
x サーバの起動方法は以下を参照してください。

- Windows→[こちら](./README.xserver.md#Windows-の場合)
- Linux→

## step3'. GUI なしでテトリスプログラムを実行

GUI の起動が必要ない場合には先程立ち上げたコンソールかたコンテナ内の環境変数`QT_QPA_PLATFORM`を`offscreen`に設定します。

```
export QT_QPA_PLATFORM=offscreen
```

## step4. テトリスプログラムの実行

`/tetris`にいることを確認し、以下を実行

```
python start.py
```

テトリスの画面が表示されれば無事完了です。

## コンテナの停止

ホスト OS 側の tetris ディレクトリから以下を実行する。

```
docker-compose stop
```

この場合には、コンテナは削除されずに残ります。

## コンテナの停止、削除

ホスト OS 側の tetris ディレクトリから以下を実行する。

```
docker-compose down
```

この場合には、コンテナは削除されます。
