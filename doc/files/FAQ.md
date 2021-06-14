# FAQ

## level1ではブロックの順番は固定とのことだが、具体的にはどんな順番になっているのか
`level1（ブロックの順番固定）`の場合は、[ブロックのindex値](https://github.com/seigot/tetris_game/blob/master/doc/files/block_controller.md#ブロック情報)が`1→2→3→4→5→6→7→1→...`の順番に出現します。

## python実行環境について

|  環境  |  環境構築手順  |
| ---- | ---- |
|  ubuntu18.04,20.04  |  [こちら](https://github.com/seigot/tetris_game/blob/master/doc/files/install_ubuntu.md)  |
|  Mac  |  [こちら](https://github.com/seigot/tetris_game/blob/master/doc/files/install_mac.md)  |
|  Windows+Docker  |  [こちら](https://github.com/seigot/tetris_game/blob/master/docker/README.md)  |
|  JetsonNano  |  （動作未確認だがおそらく動くはず）  |
|  RaspberryPi  |  （動作未確認だがおそらく動くはず）  |
|  Windows+GoogleChrome+ubuntu-free-online-linux  |  [chrome webstore URL](https://chrome.google.com/webstore/detail/ubuntu-free-online-linux/pmaonbjcobmgkemldgcedmpbmmncpbgi?hl=ja)  |
|  AWS  |  EC2 4CPU 8GBメモリ 20GBストレージ、GPU環境で動作確認済（課金に注意）  |

## `Windows+GoogleChrome+ubuntu-free-online-linux`の環境構築について

>google chrome上でubuntu serverを実行する<br>
・[google chrome用のubuntu online server拡張プラグイン](https://chrome.google.com/webstore/detail/ubuntu-free-online-linux/pmaonbjcobmgkemldgcedmpbmmncpbgi)をインストール<br>
・選択肢のうち、"Xubuntu"を選択（ubuntu18.04かつ軽量なものが良さそう）<br>
・serverにログインして以下を実施<br>
　・Desktop上で右クリックし"Open Terminal Here"を選択<br>
　・terminal上で以下コマンドを実行<br>
```
sudo apt install −y git
git clone http://github.com/seigot/tetris_game
cd tetris game
bash doc/files/install_ubuntu.sh
```
  
--> installが成功すればOK

```
bash start.sh
```

--> テトリスが表示されればOK

## `sudo apt install`時に`E: ロック /var/lib/dpkg/lock-frontend が取得できませんでした - open (11: リソースが一時的に利用できません)`のエラーが出る

[こちらのサイト](https://marginalia.hatenablog.com/entry/2019/07/03/133854)参照<br>
以下で解決するはず

```
$ sudo rm /var/lib/apt/lists/lock
$ sudo rm /var/lib/dpkg/lock
$ sudo rm /var/lib/dpkg/lock-frontend
```

## サンプルプログラム（`bash start.sh -s y`で動くやつ）の中身はどうなってるのか
[こちら](https://github.com/seigot/tetris_game/blob/master/doc/files/block_controller_sample.md)で解説

## スコアアタック時の動作PC環境について

2021/7時点で以下を想定しています。<br>
PC: [ZBOX Magnux en52060](https://www.zotac.com/jp/product/mini_pcs/magnus-en52060v#spec)<br>

```
- OS : ubuntu18.04
- CPU: Intel Core i5
- Memory: 16GB
- NVIDIA GeForce RTX 2060
```

```
- pythonバージョンは以下
$ python3 --version 
Python 3.6.9 
$ python3 -c 'import torch; print(torch.__version__) ' 
1.4.0
```

ソフト環境は以下スクリプトで構築しています。(散らかっててすみません)<br>
[auto_setup_for_Linux.sh](https://github.com/seigot/tetris_game/blob/master/docker/auto_setup_for_Linux.sh)<br>

## Windows Docker install時のトラブルシューティング

```
①BIOSのCPU関連設定
　Docker for Windowsをインストールして起動すると、
　「An error occurred Hardware assisted virtualization and data execution protection
　  must be enabled in the BIOS.」と出てくる。
　⇒ 以下サイトの「1. BIOS設定の確認」をすることでエラー解消
　　 https://qiita.com/LemonmanNo39/items/b1b104e7fb609464727b

②WSLのインストール
　Dockerを起動したところ、①は解消されたが以下のメッセージが表示される。
　「WSL2 installation is incomplete」
　⇒ 以下を参考にしてWSL2をインストールする。
　　 https://docs.microsoft.com/ja-jp/windows/wsl/install-win10
　　 ※ Docker for Windowsをインストールするときに一緒にWSLも入れられたっぽい
 
③ Dockerイメージの取得途中で死ぬ
自宅のネットワーク回線が貧弱なのか、下記コマンドでイメージを持ってくる途中で止まる。
docker run -p 6080:80 --shm-size=512m seigott/tetris_game_docker

⇒ Docker for Windowsを起動して、ウィンドウ上部の"歯車(設定)ボタン"、"Docker Engine"をクリック。
　 configファイルに、「"max-concurrent-download":1」を追記。
　※これでデフォルトの3並列ダウンロードが直列になる
 
④ Dockerイメージの取得途中で、「docker: unauthorized authentication required」と出て死ぬ
⇒ windows power shellを管理者権限で実行してコマンドを叩くといけたっぽい。
```

## pytorch v1.4 インストール方法

ubuntu18.04環境では、以下のようにしてインストールできることを確認済<br>
[pytorch v1.4 インストール済のDocker環境（お試し版）](https://github.com/seigot/tetris_game/blob/master/docker/README.md)を作成しました。追加で必要なものがあればDockerfileを更新して下さい。

```
function install_torch(){
    ### pytorch from pip image (v1.4)
    sudo apt-get install -y libopenblas-base libopenmpi-dev
    sudo apt-get -y install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
    
    #python -m pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp27-cp27mu-linux_x86_64.whl
    python -m pip install torchvision==0.2.2
    pip3 install torch==1.4.0 torchvision==0.2.2
    pip install 'pillow<7'
}
```
## Dockerはコンテナ終了の度にデータが消えて手間が掛かるので何とかしたい
[WSL(Windows Subsystem for Linux)を使う場合](https://github.com/seigot/tetris_game/blob/master/doc/files/install_windows_wsl.md)の手順を用意しました。<br>
（kyadさんありがとうございます）<br>
<br>
追記：cygwin環境構築手順<br>
[isshy-youさんによる`Cygwin Install for tetris_game`構築手順](https://github.com/isshy-you/tetris_game/wiki/Cygwin-Install-for-tetris_game)<br>
（isshy-youさんありがとうございます）

### 以下、順次追記
