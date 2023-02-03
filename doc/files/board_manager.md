>本ページでは、[ボード管理用プログラム](https://github.com/seigot/tetris/blob/master/game_manager/board_manager.py)について説明を追記頂ける方を募集しています。<br>
>説明の追記方法は、[`Pull Requestを送る`](https://github.com/seigot/tetris#pull-requestを送るoptional)を参照下さい。<br>

# ボード管理用プログラムについて

ここでは [board_manager.py](../../game_manager/board_manager.py) の解説を行う。

このファイルにはテトリミノの 形状、回転方向を担う Shape Class と、画面ボードの各種処理を行う BoardData Class がある。
block_controller.py 側からもよく使うメソッドが多いので解説を書いておく。

## Shape Class

ここでは、[board_manager.py](../../game_manager/board_manager.py) の Shape Class について解説する。
Shape Class は下記テトリミノの形状について保持しているクラスである。

|     |  ShapeI  |  ShapeL  |  ShapeJ  |  ShapeT  |  ShapeO  |  ShapeS  |  ShapeZ  |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| index値 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 
| 初期形状 | ![Screenshot](../pics/ShapeI.png) | ![Screenshot](../pics/ShapeL.png) | ![Screenshot](../pics/ShapeJ.png) | ![Screenshot](../pics/ShapeT.png) | ![Screenshot](../pics/ShapeO.png) | ![Screenshot](../pics/ShapeS.png) | ![Screenshot](../pics/ShapeZ.png) | 
| １周の回転に必要な回数 | 2 | 4 | 4 | 4 | 1 | 2 | 2 | 

詳細は、[ブロック情報](block_controller.md#ブロック情報) に詳しく記載がある。

### getRotatedOffsets method

getRotatedOffsets メソッドは テトリミノ形状を回転した基準座標からの相対座標を返します。
引数として direction テトリミノ回転方向を指定する。

[ブロック操作の基準点](block_controller.md#ブロック操作の基準点)の形状番号を direction で指定すると、ここで書かれている相対座標が返される。

### getCoords method

getCoords メソッドは さらに x,y を引数に持ち、x,y を基準点に置いたテトリミノの絶対座標を返す。

### getBoundingOffsets method

getBoundingOffsets メソッドはテトリミノの基準点からの最大 x, y 最小x, y を取得する。
direction で回転方向を指定し、minX, maxX, minY, maxY が返ってくる。






## BoardData Class

ここでは、[board_manager.py](../../game_manager/board_manager.py) の BoardData Class について解説する。
BoardData Class は画面ボードに配置されているテトリミノ情報が格納されており、画面ボードの処理はこの Class において行う。

### 画面ボード情報取得系メソッド

#### getData method

現状の画面ボードデータを返す。一次元配列である。



### テトリミノ情報取得関係メソッド

#### getDataWithCurrentBlock method

画面ボードデータコピーし動いているテトリミノデータを付加し、その画面ボードデータを返す。


#### getValue method

引数として x,y を指定し画面ボード上のその座標のテトリミノの有無(およびその種類)を返す。
返り値は Shape Class の index値。何もないなら 0。



#### getCurrentShapeCoord method

direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置した場合の座標配列を返す



#### getShapeListLength method

予告テトリミノ配列の長さを返す



#### getShapeDataFromShapeClass method

テトリミノクラスデータ, テトリミノ座標配列、テトリミノ回転種類を返す

引数

    ShapeClass ... Shape のオブジェクト

返値

    ShapeClass ... Shape のオブジェクト
    ShapeIdx ... Shape Class の index値
    ShapeRange ... Shape Class の回転方向配列。Oミノなら (0,) Lミノなら (0,1,2,3)。



#### getShapeData method

テトリミノの index 番号から shape オブジェクトを返す

引数

    ShapeNumber ... テトリミノの index

返値

    ShapeClass ... Shape のオブジェクト





### HOLD 関係

#### getholdShapeData method

ホールドしている Shape オブジェクトを返す





### ART 関係

#### getcolorTable method

colorTableを返す



#### getnextShapeIndexListDXY method

index を指定して、nextShapeIndexList の d, x, y を返す

引数

    index

返値

    d
    x
    y






### ゲームターン進行関係メソッド


#### mergePiece method

画面ボードに固着したテトリミノを書き込む


#### getNewShapeIndex method

次のテトリミノの index 取得

#### createNewPiece method

画面ボード上に新しいテトリミノの配置。配置できなかったら False を返す。


#### tryMoveNext method

direction 方向で x,y へ2回配置して動かせなかったら Reset


#### exchangeholdShape method

HOLD 入れ替え

#### removeFullLines method

画面ボードの消去できるラインを探して消去し、画面ボードを更新、そして消した Line を返す








### テトリミノ配置確認関係メソッド

#### tryMoveCurrent method

テトリミノを direction 方向で x,y に動かせるかどうか確認する
動かせない場合 tryMove メソッドを使い False を返す


#### tryMove method

direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置可能か判定する
配置できない場合 False を返す

#### moveDown method

テノリミノを1つ落とし消去ラインとテトリミノ落下数を返す

#### dropDown method

テトリミノを一番下まで落とし消去ラインとテトリミノ落下数を返す

#### moveLeft method

左へテトリミノを1つ動かす
失敗したら False を返す

#### moveRight method

右へテトリミノを1つ動かす
失敗したら False を返す

#### rotateRight method

右回転させる
失敗したら False を返す


#### rotateLeft method

左回転させる
失敗したら False を返す







### 初期配置関係メソッド


#### clear method

画面ボードと現テトリミノ情報をクリア

#### addobstacle method

初期障害物配置。Level3 Level4 用。



