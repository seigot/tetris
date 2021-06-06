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

### 以下、順次追記
