<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Tetris](#tetris)
  - [実行環境準備](#%E5%AE%9F%E8%A1%8C%E7%92%B0%E5%A2%83%E6%BA%96%E5%82%99)
      - [Mac環境](#mac%E7%92%B0%E5%A2%83)
      - [Ubuntu/JetsonNano環境](#ubuntujetsonnano%E7%92%B0%E5%A2%83)
      - [windows環境](#windows%E7%92%B0%E5%A2%83)
      - [docker環境](#docker%E7%92%B0%E5%A2%83)
  - [実行方法](#%E5%AE%9F%E8%A1%8C%E6%96%B9%E6%B3%95)
  - [ファイル構成](#%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E6%A7%8B%E6%88%90)
      - [ファイル一覧](#%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E4%B8%80%E8%A6%A7)
      - [詳細](#%E8%A9%B3%E7%B4%B0)
  - [手動操作](#%E6%89%8B%E5%8B%95%E6%93%8D%E4%BD%9C)
  - [スコアアタック用サンプルコード](#%E3%82%B9%E3%82%B3%E3%82%A2%E3%82%A2%E3%82%BF%E3%83%83%E3%82%AF%E7%94%A8%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AB%E3%82%B3%E3%83%BC%E3%83%89)
  - [Play rules](#play-rules)
    - [Score](#score)
    - [game level](#game-level)
  - [コード作成のはじめかた](#%E3%82%B3%E3%83%BC%E3%83%89%E4%BD%9C%E6%88%90%E3%81%AE%E3%81%AF%E3%81%98%E3%82%81%E3%81%8B%E3%81%9F)
    - [本リポジトリのfork](#%E6%9C%AC%E3%83%AA%E3%83%9D%E3%82%B8%E3%83%88%E3%83%AA%E3%81%AEfork)
    - [実行](#%E5%AE%9F%E8%A1%8C)
    - [自リポジトリのバイナリを公式リリースする](#%E8%87%AA%E3%83%AA%E3%83%9D%E3%82%B8%E3%83%88%E3%83%AA%E3%81%AE%E3%83%90%E3%82%A4%E3%83%8A%E3%83%AA%E3%82%92%E5%85%AC%E5%BC%8F%E3%83%AA%E3%83%AA%E3%83%BC%E3%82%B9%E3%81%99%E3%82%8B)
    - [本リポジトリの最新バージョン取り込み](#%E6%9C%AC%E3%83%AA%E3%83%9D%E3%82%B8%E3%83%88%E3%83%AA%E3%81%AE%E6%9C%80%E6%96%B0%E3%83%90%E3%83%BC%E3%82%B8%E3%83%A7%E3%83%B3%E5%8F%96%E3%82%8A%E8%BE%BC%E3%81%BF)
    - [Pull Requestを送る（Optional）](#pull-request%E3%82%92%E9%80%81%E3%82%8Boptional)
    - [FAQ](#faq)
  - [参考](#%E5%8F%82%E8%80%83)
  - [今後の課題](#%E4%BB%8A%E5%BE%8C%E3%81%AE%E8%AA%B2%E9%A1%8C)
    - [次のブロックのランダム性](#%E6%AC%A1%E3%81%AE%E3%83%96%E3%83%AD%E3%83%83%E3%82%AF%E3%81%AE%E3%83%A9%E3%83%B3%E3%83%80%E3%83%A0%E6%80%A7)
    - [AI実装](#ai%E5%AE%9F%E8%A3%85)
    - [自動評価](#%E8%87%AA%E5%8B%95%E8%A9%95%E4%BE%A1)
    - [art](#art)
  - [LICENSE](#license)
  - [Finnaly](#finnaly)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Tetris

プログラミング学習を目的とした、ブロックを操作してスコアを競うゲームです。<br>
[FAQはこちら。](https://github.com/seigot/tetris/blob/master/doc/files/FAQ.md)<br>
[tutorialはこちら。](https://github.com/seigot/tetris_game_tutorial)<br>
[tetris_score_serverはこちら](https://github.com/seigot/tetris_score_server)  
[tetris_battle_serverはこちら](https://github.com/seigot/tetris_battle_server)  

## 実行環境準備

#### Mac環境

Finder→Application→Utility→Terminalから、ターミナルを起動して以下コマンドを実行する。

```
# install pyqt5 and NumPy
brew install python3
pip3 install pyqt5
pip3 install numpy
# install other packages
brew install git
```

[doc/files/install_mac.md](./doc/files/install_mac.md)に上記手順を記載

#### Ubuntu/JetsonNano環境

[doc/files/install_ubuntu.md](./doc/files/install_ubuntu.md)に手順を記載

#### windows環境

[windowsのpowershellを使ってテトリス環境を構築する場合](./doc/files/install_windows_powershell.md)の手順<br>
[WSL(Windows Subsystem for Linux)を使う場合](./doc/files/install_windows_wsl.md)の手順<br>
[Docker for Windowsを使う場合](./doc/files/install_windows.md)の手順<br>

#### docker環境

[docker/README.md](docker/README.md)に手順を記載

## 実行方法

本リポジトリを取得

```shell
cd $HOME
git clone https://github.com/seigot/tetris
```

ゲーム開始用スクリプトを実行

```shell
cd tetris
python start.py
```

![Screenshot](doc/pics/screenshot_02.png)

## ファイル構成

#### ファイル一覧

* `game_manager/game_manager.py` : ゲーム管理用プログラム
* `game_manager/board_manager.py` : ボード管理用プログラム
* `game_manager/block_controller.py` : ブロック操作用プログラム（ブロックの操作は、このファイルを更新して下さい。）
* `start.py` : ゲーム開始用コマンド

#### 詳細

以下のような構成になっています。<br>
ブロック操作用プログラムは、管理プログラムから定期的に呼び出されるので、ボード情報から次の動作を決定して下さい。 <br>

```mermaid
  graph TB

  subgraph ゲーム管理用プログラム
    B1["game_manager.py"]
    C1["board_manager.py"]
    D1["block_controller.py<br>ここで現在のブロックの動作を決定する"]
    B1 --update--> C1
    B1 --getNextMove--> D1
    D1 --NextMove--> B1
    subgraph ボード管理用プログラム
        C1
    end
    subgraph ブロック操作用プログラム
        D1
    end
  end


  subgraph ゲーム開始用コマンド
     A1[start.py] --> B1
  end
style ブロック操作用プログラム fill:#fef
```

詳細
- [ブロック操作用プログラムについての説明](doc/files/block_controller.md) <br>
- [ボード管理用プログラムについての説明](doc/files/board_manager.md) <br>
- [ゲーム管理用プログラムについての説明](doc/files/game_manager.md) <br>

## 手動操作

実行時、以下のようにオプションを与えることで、手動操作が可能です。
操作方法は、PC操作準拠とゲーム機コントローラ準拠の2種類を選択できるようにしています。

|  手動操作  |  PC操作準拠  |  ゲーム機コントローラ準拠  |
| ---- | ---- | ---- |
|  実行コマンド  |  python start.py -m keyboard  |  python start.py -m gamepad  |
|  *up* key  |  回転  |  落下  |
|  *left* key  |  左に移動  |  左に移動  |
|  *right* key   |  右に移動  |  右に移動  |
|  *m* key  |  下に移動  |  下に移動  |
|  *space* key  |  落下  |  回転  |
|  *P* key  |  Pause  |  Pause  |
|  *c* key  |  hold  |  hold  |

## スコアアタック用サンプルコード

実行時、以下のようにオプションを与えることで、スコアアタック用サンプルコードの実行が可能です。<br>
サンプルコードについて[ブロック操作用サンプルプログラム](https://github.com/seigot/tetris/blob/master/doc/files/block_controller_sample.md)を参照下さい。<br>

```shell
python start.py -m sample
```

## Play rules

制限時間内の獲得スコアを評価します。

### Score

加点

|  項目  |  得点  |  備考  |
| ---- | ---- |  ---- |
|  1ライン消し  |  + 100点  |  -  |
|  2ライン消し  |  + 300点  |  -  |
|  3ライン消し  |  + 700点  |  -  |
|  4ライン消し  |  + 1300点  |  -  |
|  落下ボーナス  |  + 落下したブロック数を得点に加算  |  -  |

減点

|  項目  |  得点  |  備考  |
| ---- | ---- |  ---- |
|  gameover  |  - 500点  | ブロック出現時にフィールドが埋まっていたらgameover

### game level

実行時、オプションを与えることで、難易度（レベル）を指定できます。<br>

|     |  level1  |  level2  |  level3  |  level4  | 
| --- | --- | --- | --- | --- | 
|  実行方法  | python start.py | python start.py -l2  | python start.py -l3 | python start.py -l4 | 
|  制限時間  |  180秒  |  180秒  |  180秒  |  180秒  | 
|  ブロックの順番  |  固定(1-7まで順番に繰り返し)  |  ランダム  |  ランダム  |  ランダム  |
|  フィールドの初期ブロック  |  なし  |  なし  |  あり  |  あり  | 
|  フレーム更新頻度  |  約1秒  |  約1秒  |  約1秒  |  約0.001秒  | 
|  備考  |  -  |  -  |  -  |  -  | 

[各レベルの参考スコア](doc/files/reference_score.md)

## コード作成のはじめかた

### 本リポジトリのfork

まず、Githubアカウントを取得して本リポジトリを自リポジトリにforkして下さい。

> リポジトリのフォークの例 <br>
> 
> 0. GitHubアカウントを作成/ログインする。 <br>
> 1. GitHub で、[https://github.com/seigot/tetris](https://github.com/seigot/tetris)リポジトリに移動します <br>
> 2. ページの右上にある [Fork] をクリックします。 <br>
> 参考：[リポジトリをフォークする](https://docs.github.com/ja/get-started/quickstart/fork-a-repo#forking-a-repository) <br>

その後、自リポジトリにforkした`tetris`をローカルマシンに取得して下さい。

```
cd ~
git clone https://github.com/<yourname>/tetris   # "<yourname>"さん（yourname=自分のアカウント名に読みかえて下さい）のリポジトリを取得する場合
git clone https://github.com/seigot/tetris       # このリポジトリを取得する場合
```

既に`tetris`が存在しており、これを削除したい場合は`rm -f`を実行して下さい。

```
# ubuntu/mac等の場合
sudo rm -rf tetris

# windows powershellの場合
Remove-Item -Recurse -Force tetris
```

取得後はソースコード変更、変更リポジトリに反映する等してアップデートを進めて下さい。

### 実行

`実行方法`を参考に実行環境の構築をして下さい。<br>
環境構築の完了後、ブロック操作用プログラム[`block_controller.py`](https://github.com/seigot/tetris/blob/master/game_manager/block_controller.py)を更新していってください。<br>

### 自リポジトリのバイナリを公式リリースする

提出時、自リポジトリのバイナリを公式リリースする場合は、Githubリリースの機能を使うと簡単なのでお勧めです。

> 自リポジトリのコードを提出（バイナリリリース）する場合の手順参考 <br>
> [リポジトリのリリースを管理する](https://docs.github.com/ja/free-pro-team@latest/github/administering-a-repository/managing-releases-in-a-repository) <br>
> 7.オプションで、コンパイルされたプログラムなどのバイナリファイルをリリースに含めるには、ドラッグアンドドロップするかバイナリボックスで手動で選択します。 <br>

### 本リポジトリの最新バージョン取り込み

今後、本リポジトリもバージョンアップしていく予定です。<br>
本リポジトリのバージョンアップを取り込む場合は、forkしたリポジトリにて以下を実行して下さい。<br>

※追記 2021/5より、Github UI上から操作可能になったようです。<br>
[GitHub新機能「Fetch upstream」使ってみた！　1クリックで親リポジトリに追従（同期）](https://note.com/llminatoll/n/n423296287697)<br>

```
git checkout master                                        # ローカルのmainブランチに移動
git remote add upstream https://github.com/seigot/tetris  # fork元のリポジトリをupstream という名前でリモートリポジトリに登録（名前はなんでもいい。登録済みならスキップ）
git fetch upstream                                         # upstream から最新のコードをfetch
git merge upstream/master                                  # upstream/main を ローカルのmaster にmerge
git push                                                   # 変更を反映
```

参考：[github で fork したリポジトリで本家に追従する](https://please-sleep.cou929.nu/track-original-at-forked-repo.html)

### Pull Requestを送る（Optional）

本リポジトリへ修正リクエストを送ることが可能です。詳しくは参考をご参照下さい。<br>
<br>
※追記　Pull Request練習用リポジトリを作成しました。<br>
[test_pull_request](https://github.com/seigot/test_pull_request)<br>
<br>
解説図:<br>

![Git Commentary](doc/pics/20230115_Git_Commentary.png)

参考：<br>
[GitHub-プルリクエストの作成方法](https://docs.github.com/ja/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request)<br>
[[実践] はじめてのPull Requestをやってみよう](https://qiita.com/wataryooou/items/8dce6b6d5f54ab2cef04)<br>
[【GitHub】Pull Requestの手順](https://qiita.com/aipacommander/items/d61d21988a36a4d0e58b)<br>

### FAQ

[doc/files/FAQ.md](doc/files/FAQ.md)を参照下さい。

## 参考

[https://github.com/LoveDaisy/tetris_game](https://github.com/LoveDaisy/tetris_game) <br>
[https://github.com/seigot/tetris_game (2021.12時点まで使用)](https://github.com/seigot/tetris_game)<br>
[http://zetcode.com/gui/pyqt5/tetris/](http://zetcode.com/gui/pyqt5/tetris/)<br>
[テトリスの歴史を「ブロックが落ちるルール」の進化から学ぶ](https://gigazine.net/news/20191116-tetris-algorithm/)<br>

## 今後の課題

### 次のブロックのランダム性

次のブロックのランダム性は、現在はrandom関数の出力に依存しています。<br>
しかし、[こちらの記事](https://gigazine.net/news/20191116-tetris-algorithm/)によると選択方式が色々ありそうです。<br>
有識者の方からアドバイス頂けると助かります。<br>

* 参考：次のブロック選択処理 [game_manager.py](game_manager/game_manager.py)

```
nextShapeIndex = np_randomShape.random.randint(1, 8)
```

### AI実装
[AIについて](doc/files/ai.md)

### 自動評価
スコアアタック用サーバ  
https://github.com/seigot/tetris_score_server

### art
[artについて](doc/files/art.md)

## LICENSE

[MIT LICENSE](LICENSE)

## Finnaly

~ HAVE FUN ~
