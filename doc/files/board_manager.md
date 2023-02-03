>本ページでは、[ボード管理用プログラム](https://github.com/seigot/tetris/blob/master/game_manager/board_manager.py)について説明を追記頂ける方を募集しています。<br>
>説明の追記方法は、[`Pull Requestを送る`](https://github.com/seigot/tetris#pull-requestを送るoptional)を参照下さい。<br>

# ボード管理用プログラムについて

## Shape Class

ここでは、[board_manager.py](../../game_manager/board_manager.py) の Shape Class について解説する。
Shape Class は下記テトリミノの形状について保持しているクラスである。

|     |  ShapeI  |  ShapeL  |  ShapeJ  |  ShapeT  |  ShapeO  |  ShapeS  |  ShapeZ  |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| index値 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 
| 初期形状 | ![Screenshot](../pics/ShapeI.png) | ![Screenshot](../pics/ShapeL.png) | ![Screenshot](../pics/ShapeJ.png) | ![Screenshot](../pics/ShapeT.png) | ![Screenshot](../pics/ShapeO.png) | ![Screenshot](../pics/ShapeS.png) | ![Screenshot](../pics/ShapeZ.png) | 
| １周の回転に必要な回数 | 2 | 4 | 4 | 4 | 1 | 2 | 2 | 

詳細は、[ブロック情報](block_controller.md#ブロック情報) に詳しく記載があります。

### getRotatedOffsets method

getRotatedOffsets メソッドは テトリミノ形状を回転した基準座標からの相対座標を返します。
引数として direction テトリミノ回転方向を指定します。

[ブロック操作の基準点](block_controller.md#ブロック操作の基準点)の形状番号を direction で指定すると、ここで書かれている相対座標が返されます。

### getCoords method

getCoords メソッドは さらに x,y を引数に持ち、x,y を基準点に置いたテトリミノの絶対座標を返します。

### getBoundingOffsets method

getBoundingOffsets メソッドはテトリミノの基準点からの最大 x, y 最小x, y を取得します。
direction で回転方向を指定し、minX, maxX, minY, maxY が返ってきます。

## BoardData Class

ここでは、[board_manager.py](../../game_manager/board_manager.py) の BoardData Class について解説する。

### getData method

現状の画面ボードデータを返します。一次元配列です。

_執筆中_
