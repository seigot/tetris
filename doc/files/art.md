# artについて

本ドキュメントではいわゆるテトリスアートを作成する取り組みについて情報集約する

# 概要

通常、テトリスは上から落ちてくるブロック（テトリミノ）を操作して、横列を埋めて消していくようにして遊ぶ。  
テトリスアートとはブロックを消すことを必ずしも目的とせず、ブロックでフィールドに模様を描くことを目的とする。  
詳しくはgoogle検索をしてみて下さい。

# 取り組み方

まず、ブロックでフィールドに描きたい模様をイメージする。  
模様がイメージできたらブロックを操作する。  
ブロックを操作する際、各ブロックの色や出現順序(index)などをconfigファイルから調整できるようにしている。  
以下はconfigファイルの説明である。  

art用configファイルサンプル  
[config/art_config_sample.json](https://github.com/seigot/tetris/blob/master/config/art_config_sample.json)  

```
{
  // 各ブロックの色を調整する。色についてはカラーコードを参照下さい。
  "color": {
    "shapeI": "0xCC6666",
    "shapeL": "0x66CC66",
    "shapeJ": "0x6666CC",
    "shapeT": "0xCCCC66",
    "shapeO": "0xCC66CC",
    "shapeS": "0x66CCCC",
    "shapeZ": "0xDAAA00"
  },
  // 各ブロックの出現順序(index)/direction/x/yを調整する。
  // 行を追加することが可能であり、追加すると記載に応じたブロックが出現する。
  "block_order": [ 
      [1,0,0,1],
      [2,0,0,1],
      [3,0,0,1],
      [4,0,0,1],
      [5,0,0,1],
      [6,0,0,1],
      [7,0,0,1]
  ]
}
```

実行方法

```
python start.py -l1 -m art --art_config_filepath config/art_config_sample.json
```

独自のart用configファイル(`xxx.json`)を作成する場合はサンプルをコピーして使用してください。  

```
# 事前にサンプルをコピーしておく(art_config_sample.json --> xxx.json)
# 実行
python start.py -l1 -m art --art_config_filepath config/xxx.json
```
