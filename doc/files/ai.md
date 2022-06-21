## AIについて（準備中）

- sample1: CNNベースの機械学習（仮）
- sample2: DQNベースの機械学習（仮）

学習

```
python start.py -m train_sample -d 1 -l 2 -t -1
```

推論

```
python start.py -m predict -l 2 --predict_weight sample_weight/dqn1/sample_weight
```


```
pip install -r requirements.txt 
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
