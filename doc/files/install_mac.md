# 実行環境(Macの場合)

Finder→Application→Utility→Terminalから、ターミナル画面を起動して以下コマンドを実行する。<br>

```
# install pyqt6 and NumPy
brew install python3
brew install pyqt6
brew install numpy
#pip3 install pyqt5
#pip3 install numpy
# install other packages
brew install git
```

# Mac (M1) Installation Instructions

For Mac users with an M1 chip, follow these steps to set up your environment:

1. Open the terminal without Rosetta 2.
2. Create a virtual environment using the native Python:
   ```
   /usr/bin/python3 -m venv env
   source env/bin/activate
   ```
3. Upgrade pip:
   ```
   pip install --upgrade pip
   ```
4. Install PyQt6:

Mac(M1)環境等では`pip3 install pyqt5`でエラーになることがある。`brew`を使うことで解決することがある。  
[Mac(M1)にpip3 install pyqt5でpyqt5をインストールしようしようとするとエラーになる（Preparing metadata (pyproject.toml) ... error）](https://qiita.com/seigot/items/c779d187982268cf8b12)  

```
brew install pyqt5
```

`pytorch`を使う場合は以下のようにすればOK  
[MacBook Pro (Apple Silicon, M1 PRO, 2021) でPython開発環境を整える #49](https://github.com/seigot/tetris/issues/49)  
[AIについて]([https://github.com/seigot/tetris/blob/master/doc/files/install_mac.md](https://github.com/seigot/tetris/blob/master/doc/files/ai.md)  

```
# install
pip3 install -U pip
pip3 install torch
pip3 install tensorboardX
python3 -c "import torch"
# 実行例 (詳細は"AIについて"リンク参照)
python3 start.py -m train_sample -d 1 -l 2 -t -1
python3 start.py -m predict_sample -l 2 --predict_weight weight/DQN/sample_weight.pt
```

もし他にも足りないことがあれば`pull request`頂けると助かります

