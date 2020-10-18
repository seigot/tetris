# Tetris Game

プログラム学習を目的とした、テトリスを操作してスコアを競うゲームです

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
* `tetris_controller.py` : ゲーム制御用プログラム（テトリスの操作は、このファイルを編集して下さい。）
* `start.sh` : ゲーム開始用スクリプト

#### 詳細

記載予定

#### サンプルコード

記載予定

#### How to play manually

[こちらのプログラム](tetris_manager/tetris_controller_manual_sample.py)を参考に、手動操作が可能です

|  *up* key  |  回転  |
|  *left* key  |  左に移動  |
|  *right* key   |  右に移動  |
|  *m* key  |  下に移動  |
|  *space* key  |  落下  |
|  *P* key  |  pause  |

#### docker環境

[docker/README.md](docker/README.md) にdocker環境構築手順を記載

# Play rules

制限時間内の獲得スコアを評価することが可能です

## Score

加点

|  項目  |  得点  |
| ---- | ---- |
|  1ライン消し  |  + 40点  |
|  2ライン消し  |  + 100点  |
|  3ライン消し  |  + 300点  |
|  4ライン消し  |  + 1200点  |
|  落下ボーナス  |  + 落下したブロック数を得点に加算  |

減点

|  項目  |  得点  |
| ---- | ---- |
|  gameover  |  - xxx点  |

## game level

|     |  level0  |  level1  |  level2  |  level3  | 
| --- | --- | --- | --- | --- | 
|  制限時間  |  なし  |  300秒  |  300秒  |  300秒  | 
|  次のブロック  |  固定  |  固定  |  ランダム  |  ランダム  | 
|  初期ブロック  |  なし  |  なし  |  なし  |  あり  | 
|  実行方法  | bash start.sh | bash start.sh -l1 | bash start.sh -l2  | bash start.sh -l3 | 
|  フレーム更新頻度  |  約1秒  |  約1秒  |  約1秒  |  約1秒  | 

# 参考

[https://github.com/LoveDaisy/tetris_game](https://github.com/LoveDaisy/tetris_game)

# LICENSE

[LICENSE](LICENSE)

# Finnaly

~ HAVE FUN ~
