# 実行環境(Macの場合)

Finder→Application→Utility→Terminalから、ターミナル画面を起動して以下コマンドを実行する。<br>

```
# install pyqt5 and NumPy
brew install python3
pip3 install pyqt5
pip3 install numpy
# install other packages
brew install git
```

Mac(M1)環境等では`pip3 install pyqt5`でエラーになることがある。`brew`を使うことで解決することがある。  
[Mac(M1)にpip3 install pyqt5でpyqt5をインストールしようしようとするとエラーになる（Preparing metadata (pyproject.toml) ... error）](https://qiita.com/seigot/items/c779d187982268cf8b12)  

```
brew install pyqt5
```

もし他にも足りないことがあれば教えて頂けると助かります

