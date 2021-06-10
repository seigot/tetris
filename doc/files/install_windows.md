# 実行環境(windowsの場合)
docker for windowsを使います。<br>

### step0. DockerDesktopのインストール 

以下を参照する。<br>
[DockerDeskop](https://docs.docker.com/docker-for-windows/install) <br>
[Windows10用Windows Subsystem for Linuxのインストール ガイド](https://docs.microsoft.com/ja-jp/windows/wsl/install-win10)

以下にインストール時のトラブルシューティングをまとめました。<br>
[FAQ/Windows Docker install時のトラブルシューティング](https://github.com/seigot/tetris_game/blob/master/doc/files/FAQ.md#windows-docker-install%E6%99%82%E3%81%AE%E3%83%88%E3%83%A9%E3%83%96%E3%83%AB%E3%82%B7%E3%83%A5%E3%83%BC%E3%83%86%E3%82%A3%E3%83%B3%E3%82%B0)

### step1. パワーシェル上でコンテナを起動する<br>

`Windowsボタン`から`Windows PowerShell`を起動してterminalから以下実行する。<br>
コンテナのダウンロード〜起動が完了するまで少し待つ。<br>

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

注意）パワーシェルを閉じたり電源OFFするとコンテナも終了します。作成中データが失われないよう注意してください。
