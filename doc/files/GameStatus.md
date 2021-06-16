## `GameStatus`データ構造

[`block_controller.py`](https://github.com/seigot/tetris_game/blob/master/block_controller.py)内の`def GetNextMove(self, nextMove, GameStatus)`で扱う`GameStatus`は辞書型データです。<br>
ブロックの動きを決定するための参考データとして以下を格納しています。

```
* field_info : フィールド情報
* block_info : ブロック情報
* judge_info : 審判情報
* debug_info : デバッグ情報
```

以下、各情報の詳細について記載します。

## field_info

以下を格納しています。

```
* 'backboard': フィールドのデータ
* 'height': フィールドの高さ
* 'width' : フィールドの幅
* 'withblock' :  フィールド+ブロックの位置を合わせたデータ
```

具体的には以下のようなデータとなっています。実際に出力してみると分かり易いと思います。

```
'field_info': {'backboard': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               'height': 22,
               'width': 10,
               'withblock': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
```

## block_info

以下を格納しています。

```
'currentDirection':        現在のブロックが初期形状から何回回転した形状か
'currentShape': {
       'class':            現在のブロックを管理しているクラスのアドレス
       'direction_range':  現在のブロックが回転できる回数
       'index':            現在のブロックのindex値
},
'currentX':                現在のブロック操作の基準点(x座標)
'currentY':                現在のブロック操作の基準点(y座標)
'nextShape': {
        'class':           次のブロックを管理しているクラスのアドレス
        'direction_range': 次のブロックが回転できる回数
        'index' :          次のブロックのindex値
}
```

具体的には以下のようなデータとなっています。実際に出力してみると分かり易いと思います。

```
'block_info': {'currentDirection': 0,
                'currentShape': {'class': <board_manager.Shape object at 0x11743c7f0>,
                                 'direction_range': (0, 1),
                                 'index': 1},
                'currentX': 5,
                'currentY': 1,
                'nextShape': {'class': <board_manager.Shape object at 0x11743cb38>,
                              'direction_range': (0, 1, 2,
                                                  3),
                              'index': 2}},
```

## debug_info

以下を格納しています。<br>
デバック用に準備している情報であり、必ずしも使う必要はないかもしれないです。

```
'dropdownscore':     落下により獲得したスコアの合計
'line_score': {      lineを消した時などに獲得できるスコア
      '1':           1line消した時の獲得スコア
      '2':           2line消した時の獲得スコア
      '3':           3line消した時の獲得スコア
      '4':           4line消した時の獲得スコア
      'gameover':    game overになった時の獲得スコア
},
'line_score_stat':   消したlineの統計データ
'linescore':         lineを消して獲得したスコアの合計
'shape_info': {
     'shapeI': {
          'color':   ShapeIの色
          'index':   ShapeIのindex
     },
     'shapeJ': {
          'color':   ShapeJの色
          'index':   ShapeJのindex
     },
     'shapeL': {
          'color':   ShapeLの色
          'index':   ShapeLのindex
     },
     'shapeNone': {
          'color':   ShapeNone(ブロックがない状態)の色
          'index':   ShapeNone(ブロックがない状態)のindex
     },
     'shapeO': {
          'color':   ShapeOの色
          'index':   ShapeOのindex
     },
     'shapeS': {
          'color':   ShapeSの色
          'index':   ShapeSのindex
     },
     'shapeT': {
          'color':   ShapeTの色
          'index':   ShapeTのindex
     },
     'shapeZ': {
          'color':   ShapeZの色
          'index':   ShapeZのindex
     }
},
'shape_info_stat':   ゲーム開始時から現時点までに出現したブロックの統計情報
```

具体的には以下のようなデータとなっています。実際に出力してみると分かり易いと思います。

```
 'debug_info': {'dropdownscore': 0,
                'line_score': {'1': 100,
                               '2': 300,
                               '3': 700,
                               '4': 1300,
                               'gameover': -500},
                'line_score_stat': [0, 0, 0, 0],
                'linescore': 0,
                'shape_info': {'shapeI': {'color': 'red',
                                          'index': 1},
                               'shapeJ': {'color': 'purple',
                                          'index': 3},
                               'shapeL': {'color': 'green',
                                          'index': 2},
                               'shapeNone': {'color': 'none',
                                             'index': 0},
                               'shapeO': {'color': 'pink',
                                          'index': 5},
                               'shapeS': {'color': 'blue',
                                          'index': 6},
                               'shapeT': {'color': 'gold',
                                          'index': 4},
                               'shapeZ': {'color': 'yellow',
                                          'index': 7}},
                'shape_info_stat': [0, 1, 0, 0, 0, 0, 0, 0]},
```

## judge_info

以下を格納しています。

```
'block_index':       現在のブロックがゲーム開始時から何番目に登場したブロックか
'elapsed_time':      ゲーム開始時からの経過時間(s)
'game_time':         制限時間(s)
'gameover_count':    ゲーム開始時から現在までにゲームオーバーになった回数
'line':              ゲーム開始時から現在までに消したラインの数
'score':             ゲーム開始時から現在までに獲得した合計スコア
```

具体的には以下のようなデータとなっています。実際に出力してみると分かり易いと思います。

```
 'judge_info': {'block_index': 1,
                'elapsed_time': 1.083,
                'game_time': 180,
                'gameover_count': 0,
                'line': 0,
                'score': 0}}
 ```
 
