# artについて

本ドキュメントではテトリスアートの取り組みについて記載する

# 概要

通常、テトリスは上から落ちてくるブロック（テトリミノ）を操作して、横列を埋めて消していくようにして遊ぶ。  
一方、テトリスアートはブロックでフィールドに模様を描くことを目的とする。（ブロックを消すことを必ずしも目的としない）  
詳しくは"テトリスアート"でgoogle検索をしてみて下さい。  

# 取り組み方

1.まず、ブロックでフィールドに描きたい模様をイメージする。  
2.次に、模様がテトリス上で実現可能かどうかを検討する。（作図などがお勧めです）  
3.実現可能であればブロックを操作して模様を作成する。
  
3.によりブロックを操作する場合、後述するconfigファイルを使い各ブロックの色や出現順序(index)などを調整できるようにしている。  

# サンプルコード

`python start.py`実行時に以下オプションを指定するとサンプルコードが実行される  
[youtube-link: tetris art sample](https://www.youtube.com/watch?v=Seh6g9_nL6o)  

`1:onigiri`

```
python start.py -l1 -m art --art_config_filepath config/art/art_config_sample1.json
```

![Screenshot](../pics/art_sample_onigiri.png)

`2:manji`

```
python start.py -l1 -m art --art_config_filepath config/art/art_config_sample2.json
```

![Screenshot](../pics/art_sample_manji.png)

`3:cartoon charactor`

```
python start.py -l1 -m art --art_config_filepath config/art/art_config_sample3.json
```

![Screenshot](../pics/art_sample_cartoon.png)

other sample 

|  name  |  --art_config_filepath  |  note  |
| ---- | ---- | ---- |
|  4:heart  |  python start.py -l1 -m art  --art_config_filepath config/art/art_config_sample4.json  |  -  |
|  5:hamburger_shop  |  python start.py -l1 -m art  --art_config_filepath config/art/art_config_sample5.json  |  -  |
|  6:parking  |  python start.py -l1 -m art  --art_config_filepath config/art/art_config_sample6.json  |  -  |
|  7:team rocket  |  python start.py -l1 -m art  --art_config_filepath config/art/art_config_sample7.json  |  -  |
|  8:happy_new_year_2023  |  python start.py -l1 -m art  --art_config_filepath config/art/art_config_sample8.json  |  -  |
|  9:taka  |  python start.py -l1 -m art  --art_config_filepath config/art/art_config_sample9.json  |  -  |
|  10:python_logo  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample10.json -d100 --BlockNumMax 500  |  -  |
|  11:yoshi  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample11.json -d100 --BlockNumMax 900  |  -  
|  12:neko  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample12.json -d100 --BlockNumMax 900  |  -  |
|  13:kyubei  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample13.json -d100 --BlockNumMax 900  |  -  |
|  14:question_block  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample14.json -d100 --BlockNumMax 900  |  -  |
|  15:coin  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample15.json -d100 --BlockNumMax 900  |  -  |
|  16:clefairy  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample16.json -d100 --BlockNumMax 900  |  -  |
|  17:jigglypuff  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample17.json -d100 --BlockNumMax 900  |  -  |
|  18:docker  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample18.json -d100 --BlockNumMax 900  |  -  |
|  19:tux  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample19.json -d100 --BlockNumMax 900  |  -  |
|  20:droid  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample20.json -d100 --BlockNumMax 900  |  -  |
|  21:jenkins  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample21.json -d100 --BlockNumMax 900  |  -  |
|  22:firefox  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample22.json -d100 --BlockNumMax 900  |  -  |
|  23:gopher  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample23.json -d100 --BlockNumMax 900  |  -  |
|  24:goomba  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample24.json -d100 --BlockNumMax 900  |  -  |
|  25:toad  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample25.json -d100 --BlockNumMax 900  |  -  |
|  26:super_star  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample26.json -d100 --BlockNumMax 900  |  -  |
|  27:fire_flower  |  python start.py -l1 -m art --art_config_filepath config/art/art_config_sample27.json -d100 --BlockNumMax 900  |  -  |
|  ...  |  -  |  -  |

contribution

|  name  |  --art_config_filepath  |  thanks  |
| ---- | ---- | ---- |
|  mario  |  python start.py -l1 -m art  --art_config_filepath config/art/art_config_sample_tanaka_2.json  |  @tanaken  |
|  ...  |  -  |  -  |

# configファイルの説明

テトリスアートを作成し易くするために各ブロックの色や出現順序(index)などをconfigファイルで調整できるようにしている。  
以下はconfigファイルの説明である。  

art用configファイルサンプル  
[config/art/art_config_sample_default.json](https://github.com/seigot/tetris/blob/master/config/art/art_config_sample_default.json)  

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

ブロックの色については以下のカラーコードを参考にしてください。  

> [色の名前とカラーコードが一目でわかるWEB色見本 原色大辞典 - HTMLカラーコード](https://www.colordic.org/)  

各ブロックの出現順序(index)/direction/x/yについては以下を参考にしてください。  

> [ブロック操作用プログラムについて](./block_controller.md)  

# 実行方法

以下のように`-m art`,`--art_config_filepath`によりモード及びconfigファイルを指定する。

```
# "art_config_sample.json"を指定して実行
python start.py -l1 -m art --art_config_filepath config/art/art_config_sample.json

# 落下速度を早くする場合は"-d500"を加えて実行
python start.py -l1 -m art --art_config_filepath config/art/art_config_sample.json -d500
```

自作したart用configファイル(`xxx.json`)を作成する場合は以下のようにファイルコピーして使用してください。  

```
# 事前にサンプルをコピーしておく(art_config_sample.json --> xxx.json)
cp config/art/art_config_sample.json config/art/xxx.json
# 実行
python start.py -l1 -m art --art_config_filepath config/art/xxx.json
```


# 作成アルゴリズムの例 
#### ◆2ドットプリンター方式  
横10マスのフィールドにテトリスミノを置いた後に残るブロックの個数は 4n-10m 個です。  
そのためテトリスアートを 4n-10m 個のドットの組み合わせに分解することで楽に作ることができます。  
汎用性が高いのは2ドット残し（32ブロック：n=8, m=3 や 52ブロック：n=13, m=5 など）。  
アートには1行あたり最大9個のドットを置けます。  
そのため基本的に各行を 2\*4 + 1ドットに分解し、余った1同士で2ドットを構成することを目指します。
```
2ドットを同じ行に残す場合、その組み合わせは恐らく2205通り（x座標10*9÷2、shape7*7）。  
2ドットを違う行に残す場合、その組み合わせは恐らく4900通り（x座標10*10、shape7*7）。  
アートを構成する2ドットを残せる出現順序 [index,direction,x,y] をあらかじめ用意すれば、config作成の自動化が可能になります。  
```


**Step1：任意の2ドットを残す出現順序をまとめる**  
`art_lib.py`：
> [https://github.com/mattshamrock/tetris/blob/art/art_lib.py](https://github.com/mattshamrock/tetris/blob/art/art_lib.py)  
xs_0000 ← 同じ行に残す場合。例えば 'xs_5294'なら、x=5にshapeindex=2（Lミノ）、x=9にshapeindex=4（Tミノ）が残る置き方。
xs_00_00 ← 違う行に残す場合。前半が1行目のx座標とshapeindex、後半が2行目のx座標とshapeindex。  
未発見の場合は False と記載  

```
art_lib.pyの現状：  
【2023/3/E時点】  
 ・'xs_0000'のうち32ブロック使用で実現可能なもの：全探索済み; 後述  
 ・'xs_0000'のうち52ブロック以上使用で実現可能なもの：手作業。不完全  
 ・'xs_00_00'のうち見つけたもの：手作業。とても不完全  
```

`art_lib.py`用の探索アルゴリズム：
> [https://github.com/mattshamrock/tetris/blob/art/artlib_search_32.py](https://github.com/mattshamrock/tetris/blob/art/artlib_search_32.py)  
各xs_0000について、32ドット（8ミノ）の出現順序をシミュレートし、指定の2ドットのみが残るかを判定します。  
枝切りをすることで探索時間を減らそうとしています。手元PCでは36時間で半分の約1200通りが出力できます。  
後述の「反転」と組み合わせると約36時間で完了します。

`art_lib.py`用の反転アルゴリズム：
> [https://github.com/mattshamrock/tetris/blob/art/artlib_reverse.py](https://github.com/mattshamrock/tetris/blob/art/artlib_reverse.py)  
横10マスのテトリスはすべての配置を左右反転させても成り立ちます。  
探索でart_lib.pyを埋めていく際、結果が出た出現順序を反転させれば探索のみで埋める約半分の時間で完了できるはずです。   
注意…基準点ズレの調整が必要な組み合わせが存在します。（例：shape:Sの形状:1 の反転とshape:Zの形状:1 は基準点が1マス違う）  


     
    
**Step2：ドットアートを描く**  
現状は手作業です  
　　


**Step3：ドットアートを数値に変換する**  
ブロックのindex値に変換します。  
Step4の入力に合わせて文字数=10のstrに変換しています。  
空白=0です。  
現状はexcelで作業しています  
　　

**Step4：art_lib.pyにある実現可能な2ドットの組み合わせを探索する**  
`art_lib.py`の組み合わせ探索アルゴリズム：
> [https://github.com/mattshamrock/tetris/blob/art/create_art.py](https://github.com/mattshamrock/tetris/blob/art/create_art.py)  
10個のindex値を2ドットの組み合わせに分解します。  
art_lib.pyでは'xs_0000'の出現順序が未発見の場合にFalseを返すことを利用して、実現可能な2ドットの組み合わせを探索できます。  
1行に奇数個のドット（空白を除く）がある場合、余ったドットのx座標とindex値も返します。これが'xs_00_00'の前半か後半のいずれかになります。
  
　  
  
**Step5：Step4の結果を調整する**  
・'xs_00_00' は余りとして返された情報を基に自力で発見し、`art_lib.py`に追加する必要があります。  
・余りとして返るドットのみで'xs_00_00'が構成されることを前提にしています。そのため各行のドット数（空白を除く）の偶奇によっては成立しません。  
・'xs_0000' において、すぐ下のマスが空白の場合に、そこにハマる形で意図しない結果になる場合があります。左右のマスにブロックがあれば引っ掛かるため回避可能な可能性が高く、その場合は順番を変更することで解決できます。  
　　

**Step6：2ドットの組み合わせをconfigファイルの出現順序に変換する**  
Step4-5で得られた 'xs_0000' と 'xs_00_00' の羅列を`art_lib.py`の`artdict`のキーとして利用し、出力します。  
現状は余計な二重括弧　[[]] が入ってきてしまいます。置換等で取り除きます。  
各出力の最後のカンマ , が不足しています。二重括弧と合わせて置換で対応します。  
　　

**Step7：configファイルに転記する**  
転記すればOK。  
色の調整は手作業です。  
　　

**今後に向けて：**  
・`art_lib.py`の抜けている部分を埋める。52ブロック（13手）以上を効率的に探索する。  
・`art_lib.py`の抜けている部分を埋める。2行に分かれるブロックを効率的に探索する。32マスを8つに分割した結果がテトリスミノの形になるか判定する、という探し方が良い気がする。  
・Step5の調整を自動化する。  
・Step2を自動化する（画像生成AI的な）  



  



# 参考
[「テトリス」の斬新すぎる遊び方が話題に。積み上げたブロックでマリオやルイージを再現!?](https://nlab.itmedia.co.jp/nl/articles/1109/13/news025.html)  
[色の名前とカラーコードが一目でわかるWEB色見本 原色大辞典 - HTMLカラーコード](https://www.colordic.org/)  
[youtube-link: tetris art sample](https://www.youtube.com/watch?v=Seh6g9_nL6o)  
