>本ページでは、[ゲーム管理用プログラム](https://github.com/seigot/tetris/blob/master/game_manager/game_manager.py)について説明を追記頂ける方を募集しています。<br>
>説明の追記方法は、[`Pull Requestを送る`](https://github.com/seigot/tetris#pull-requestを送るoptional)を参照下さい。<br>

# ゲーム管理用プログラムについて

## start.py の Option 設定
#### -m --mode モード
    default : ルールベース
    train : AI学習
    predict : AI 推論（予測実行)
    predict_sample : AI MLP 推論（予測実行)
    predict_sample2 : AI DQN 推論（予測実行)
    keyboard : キーボード操作
    gamepad : ゲームパッド操作
#### -l --game_levelレベル
    1 : Level 1 固定テトリミノ
    2 : Level 2 ランダムテトリミノ
    3 : Level 3 初期ブロックありテトリミノ
    4 : Level 4 初期ブロックありテトリミノ 0.001秒更新
#### -d --drop_interval 更新間隔
    default => 1000 (=1秒)
    ms 指定
        ※テトリミノを DROP すれば この時間待てば次のテトリミノが出るが、降下だけさせた場合は次の操作ができるまでこの時間分停止する。
        ※ Level 4指定した場合は無効
#### -t --game_time ゲーム時間
    default => 180
    秒指定
    -1 で制限なし 学習時は -1 指定推奨。
#### -r --random_seed 乱数のタネ
    整数指定。特別に必要なければ指定しない。
#### -f --resultlogjson
    default => result.json
    結果の json ファイル指定
#### --train_yaml
    default => config/default.yaml
    AI 学習推論時 (train, predict)の設定ファイル指定
#### --predict_weight
    default => outputs/latest/best_weight.pt
    AI 推論時 (predict) の学習結果 weight ファイル指定

## Game_Manager Class

ゲーム管理 Class

### 概要

最初に start.py から呼び出される。このゲームの基本的な動作はこのクラスから各クラス、メソッドを実行することによって行われる。

### 初期化

\_\_init\_\_ にて Option や初期設定値を各インスタンス変数に格納する。
さらに initUI 関数にて、Window 表示が行われる。ここで SidePanel Class, Board Class のインスタンスも生成される。

### timerEvent

_執筆中_

## SidePanel

横の予告テトリミノ、およびホールドテトリミノ描画画面 Class

_執筆中_

## Board

ゲームの主画面ボード Class

_執筆中_
