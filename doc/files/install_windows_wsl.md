# 実行環境 (Windows WSLを使う場合)

Windows WSLを使って開発環境を立ち上げる方法を説明する。
Dockerを使う方法と比較して、Dockerが不要な点と、Dockerコンテナ終了でデータが失われることを気にする必要が無い特長がある。

## Step 1. WSLのインストール 

- 以下を参照してインストールする。
  - [Windows10用Windows Subsystem for Linuxのインストール ガイド](https://docs.microsoft.com/ja-jp/windows/wsl/install-win10)

### a. WSL1でインストールする場合（簡単）

- 上記URLの「手動インストールの手順」の「手順 1 - Linux 用 Windows サブシステムを有効にする」「手順 6 - 選択した Linux ディストリビューションをインストールする」のみ実施すれば良い。
- 「手順 6 - 選択した Linux ディストリビューションをインストールする」では、Ubuntu 18.04 LTS または Ubuntu 20.04 LTS をインストールすれば良い。

### b. a以外の場合

- WSL1でもWSL2でも良い。
- Linuxのディストリビューションはどれでも良いはずであるが、Ubuntu 18.04 LTS または　Ubuntu 20.04 LTS をおすすめする。

## Step 2. Xサーバのインストール

GWSLまたはVcXsrvをインストールする。

### a. GWSLを使う場合（おすすめ）

1. スタートメニューからMicrosoft Storeアプリを起動する。
2. GWSLを検索して、インストールする。

### b. VcXsrvを使う場合

1. https://sourceforge.net/projects/vcxsrv/ からVcXsrvを入手してインストールする。

## Step 3. WSLの起動

### ターミナルを開く

- スタートメニューから、Step 1でインストールしたLinux ディストリビューション(Ubuntu 18.04 LTS など)を起動する。

### 必要なパッケージのインストール・設定（初回のみ必要）
- 下記を実行する。

```
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-pyqt5
pip3 install --upgrade pip
pip3 install numpy
sudo apt-get install -y git

echo 'export DISPLAY=localhost:0.0' >> ~/.bashrc
source ~/.bashrc
```

## Step 4. Xサーバの起動

Step 2 でインストールしたXサーバを起動する。

### a. GWSLを使う場合

1. スタートメニューからGWSLを起動する。

### b. VcXsrvを使う場合

1. スタートメニューやデスクトップのショートカットからVcXsrvを起動する。
2. 「次へ」を3回押して、「完了」を押すとXサーバが起動する。

## Step 5. tetris_gameの起動

### ソースコードを取得する（初回のみ必要）

```
git clone https://github.com/seigot/tetris_game
```

### tetris_gameを実行する
```
cd ~/tetris_game
bash start.sh
```

一度実行環境を立ち上げた後の2回目からは、Step 3以降を実行すれば良い。
