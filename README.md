# Tetris Game

テトリスを操作してスコアを競うゲームです

## 実行方法

```shell
bash start.sh
```

![Screenshot](doc/pics/screenshot_02.png)

## 実行環境

* Need python3, PyQt5 and NumPy to be installed.

```
# install pyqt5 and NumPy
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-pyqt5
pip3 install --upgrade pip
pip3 install numpy
```

## ファイル構成

* `tetris_manager/tetris_game.py` : ゲーム管理用プログラム
* `tetris_manager/tetris_model.py` : ボード管理用プログラム
* `tetris_ai.py` : ゲーム制御用プログラム（テトリスの操作は、このファイルを編集して下さい。）
* `start.sh` : ゲーム開始用スクリプト

#### 詳細

記載予定

#### docker環境

`docker/README.md` にdocker環境構築手順を記載

## How to play manually

Play just like classical Tetris Game. 
You use *up* key to rotate a shape, *left* key to move left and *right* key to move right. 
Also you can use *space* key to drop down current shape immediately and *m* key to just movedown.
If you want a pause, just press *P* key. The right panel shows the next shape.

# Play rules

記載予定

## Score

記載予定

# 参考

[https://github.com/LoveDaisy/tetris_game](https://github.com/LoveDaisy/tetris_game)

# Finnaly

~ HAVE FUN ~
