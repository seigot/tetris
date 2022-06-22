# AIについて（準備中）

# 1.環境準備
- sample のコードでは [pytorch](https://pytorch.org/get-started/locally/) を使ってAIのニューラルネットワークを構築します。  
- pytorch　については[こちら](https://pytorch.org/get-started/locally/)を参考にし、環境に合わせてインストールしてください。

### 例) WindowsのCPU動作用をインストールする場合
```
pip3 install torch torchvision torchaudio
```
### 例) MacのCPU実行用をインストールする場合
```
pip3 install torch torchvision torchaudio
```


# 2.学習と推論

* サンプルとして下記の２つのニューラルネットワークが 
[こちら](../../game_manager/machine_learning/model/deepqnet.py)
のコードに定義されています。

    * sample: DQN(Deep Q Network)を使った学習・推論
    * sample2: MLP (Multilayer perceptron)を使った学習・推論


## 2.1学習の実行
- DQN(Deep Q Network)を使う場合

```
python start.py -m train_sample -d 1 -l 2 -t -1
```

-  MLP (Multilayer perceptron)を使う場合
```
python start.py -m train_sample2 -d 1 -l 2 -t -1
```

## 2.2推論の実行

- DQN(Deep Q Network)を使う場合
```
python start.py -m predict_sample -l 2 --predict_weight sample_weight/DQN/sample_weight.pt
```

-  MLP (Multilayer perceptron)を使う場合
```
python start.py -m predict_sample2 -l 2 --predict_weight sample_weight/MLP/sample_weight.pt
```

# 3. 強化学習について
[サンプルコード1](../../game_manager/machine_learning/block_controller_train_sample.py)  　および　[サンプルコード2](../../game_manager/machine_learning/block_controller_train_sample2.py) では強化学習と呼ばれる方法でテトリスをプレイするための最適なパラメータを学習します。