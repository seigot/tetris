#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime
import random
import copy
import torch
import torch.nn as nn
import sys
sys.path.append("game_manager/machine_learning/")
import os
from tensorboardX import SummaryWriter
from collections import deque
from random import random, sample,randint
import shutil
import glob 
import numpy as np
import yaml

###################################################
###################################################
# ブロック操作クラス
###################################################
###################################################

class Block_Controller(object):
    ####################################
    # 起動時初期化
    ####################################
    board_backboard = 0
    board_data_width = 0
    board_data_height = 0
    ShapeNone_index = 0
    CurrentShape_class = 0
    NextShape_class = 0

    def __init__(self):
        self.mode = None
        self.init_train_parameter_flag = False
        self.init_predict_parameter_flag = False

    ####################################
    # Yaml パラメータ読み込み
    ####################################
    def yaml_read(self,yaml_file):
        with open(yaml_file, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg

    ####################################
    # 初期 parameter を設定
    ####################################
    def set_parameter(self,yaml_file=None,predict_weight=None):
        
        ########
        ## 保存するフォルダの作成、保存す流ファイル名の設定
        self.result_warehouse = "outputs/"
        self.latest_dir = self.result_warehouse+"/latest"
        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2":
            dt = datetime.now()
            self.output_dir = self.result_warehouse+ dt.strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs(self.output_dir,exist_ok=True)
            self.weight_dir = self.output_dir+"/trained_model/"
            self.best_weight = self.weight_dir + "best_weight.pt"
            os.makedirs(self.weight_dir,exist_ok=True)
        else:
            dirname = os.path.dirname(predict_weight)
            self.output_dir = dirname + "/predict/"
            os.makedirs(self.output_dir,exist_ok=True)

        ########
        ## Config Yaml 読み込み
        if yaml_file is None:
            raise Exception('Please input train_yaml file.')
        elif not os.path.exists(yaml_file):
            raise Exception('The yaml file {} is not existed.'.format(yaml_file))
        cfg = self.yaml_read(yaml_file)

        ####################
        # 使用する yaml ファイルを保存しておく
        shutil.copy2(yaml_file, self.output_dir)
        
        ####################
        # Tensorboard 出力フォルダ設定
        self.writer = SummaryWriter(self.output_dir+"/"+cfg["common"]["log_path"])

        ###########################
        # ログファイル設定
        ##################
        # 推論の場合
        if self.mode=="predict" or self.mode=="predict_sample" or self.mode == "predict_sample2":
            self.log = self.output_dir+"/log_predict.txt"
            self.log_score = self.output_dir+"/score_predict.txt"
            self.log_reward = self.output_dir+"/reward_predict.txt"
        ###########
        # 学習の場合
        else:
            self.log = self.output_dir+"/log_train.txt"
            self.log_score = self.output_dir+"/score_train.txt"
            self.log_reward = self.output_dir+"/reward_train.txt"
        #ログ
        with open(self.log,"w") as f:
            print("start...", file=f)
    
        #スコアログ
        with open(self.log_score,"w") as f:
            print(0, file=f)
            
        #報酬ログ
        with open(self.log_reward,"w") as f:
            print(0, file=f)
            
        # Tetris のボードに関するパラメータ
        self.height = cfg["tetris"]["board_height"]
        self.width = cfg["tetris"]["board_width"]
        
        ###########################
        # 初期化
        ##################
        ## 1エポック = 1ゲーム
        self.epoch = 0
        self.score = 0
        self.max_score = -99999
        # 獲得した報酬（毎ゲーム初期化）
        self.epoch_reward = 0
        # 削除した列数（毎ゲーム初期化）
        self.cleared_lines = 0
        # 落としたブロック数（毎ゲーム初期化）
        self.tetrominoes = 0
        # replay memory のゲーム数カウント用
        self.iter = 0 
        # １epoch で落とすブロック数の最大値
        self.max_tetrominoes = cfg["tetris"]["max_tetrominoes"]
        
        ###########################
        ### config/default.yaml でニューラルネットワークを選択
        ##################

        # MLP の場合
        print("model name: %s"%(cfg["model"]["name"]))
        if cfg["model"]["name"]=="MLP":
            self.state_dim = cfg["state"]["dim"]
            from machine_learning.model.deepqnet import MLP
            self.model = MLP(self.state_dim)
            self.initial_state = torch.FloatTensor([0 for i in range(self.state_dim)])
            self.get_next_func = self.get_next_states
            self.reward_func = self.step
            self.state = self.initial_state 

        # DQN の場合
        elif cfg["model"]["name"]=="DQN":
            from machine_learning.model.deepqnet import DeepQNetwork
            # DQNモデルのインスタンス作成
            self.model = DeepQNetwork()
            # 初期状態規定
            self.initial_state = torch.FloatTensor([[[0 for i in range(10)] for j in range(22)]])
            #各関数規定
            self.get_next_func = self.get_next_states_v2
            self.reward_func = self.step_v2
            # 報酬関連規定
            self.reward_weight = cfg["train"]["reward_weight"]

            self.state = self.initial_state

        # 推論の場合 推論ウェイトを torch で読み込み model に入れる。
        if self.mode=="predict" or self.mode=="predict_sample" or self.mode == "predict_sample2":
            print("Load {}...".format(predict_weight))
            self.model = torch.load(predict_weight)
            self.model.eval()

        # Finuetuningの場合  学習済みの weight ファイルを読み込み model に入れる。
        elif cfg["model"]["finetune"]:
            self.ft_weight = cfg["common"]["ft_weight"]
            if not self.ft_weight is None:
                self.model = torch.load(self.ft_weight)
                with open(self.log,"a") as f:
                    print("Finetuning mode\nLoad {}...".format(self.ft_weight), file=f)
                        
#        if torch.cuda.is_available():
#            self.model.cuda()
        
        ###########################
        # AI関連のパラメータ
        ##################
        #  学習バッチサイズ(学習の分割単位, データサイズを分割している)
        self.batch_size = cfg["train"]["batch_size"]
        
        # pytorch 互換性のためfloat に変換
        self.lr = cfg["train"]["lr"]
        
        # pytorch 文字列の場合はfloat に変換
        if not isinstance(self.lr,float):
            self.lr = float(self.lr)
        
        # 最大 Episode サイズ = 最大テトリミノ数
        # 1 Episode = 1 テトリミノ
        self.max_episode_size = self.max_tetrominoes
        self.episode_memory = deque(maxlen=self.max_episode_size)
        
        #EPOCH 数 (1 EPOCH = 1ゲーム)
        self.num_epochs = cfg["train"]["num_epoch"]
    
        # 勾配法(ADAM or SGD) の決定
        # ADAM の場合 
        ## 学習率も自動で調整してくれるのでSGDより賢い。スケジューラ（どの程度学習させたら減衰させるか）はいらない
        if cfg["train"]["optimizer"]=="Adam" or cfg["train"]["optimizer"]=="ADAM":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = None
        # SGD の場合 (確率的勾配降下法)
        ## 学習率を変更するスケジューラが必要
        else:
            self.momentum =cfg["train"]["lr_momentum"] 
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
            ### 減衰するタイミング
            self.lr_step_size = cfg["train"]["lr_step_size"]
            ### 学習率の減衰率
            self.lr_gamma = cfg["train"]["lr_gamma"]
            ### スケジューラ
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size , gamma=self.lr_gamma)
        
        # 勾配法(ADAM or SGD) の決定
        self.criterion = nn.MSELoss()
        
        ###########################
        # 強化学習関連のパラメータ
        ##################
        # 割引率
        ## 0~1 に設定
        ## 0.0 に近いほど直後の行動のみ評価する、1.0に近いほど過去の行動も含めて評価する
        ## 例) 割引率 0.9 で10手目で報酬 1.0 を取得した場合
        ### 10手目 1.0 , 9手目 0.9, 8手目: 0.81 ... のように過去の手を評価する
        self.gamma = cfg["train"]["gamma"]
        
        # 削除した列数とスコアのリスト
        self.score_list = cfg["tetris"]["score_list"]
        
        # 削除した列数と報酬のリスト
        self.reward_list = cfg["train"]["reward_list"]
        
        # ペナルティ値 (ゲームオーバーした時の報酬)
        self.penalty =  self.reward_list[5]
        
        # リプレイメモリ = 過去のゲームので得られたデータを貯めるバッファ
        # リプレイメモリサイズ = 何手分貯められるか
        self.replay_memory_size = cfg["train"]["replay_memory_size"]
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        
        # epsilon: epsilon-greeedy 法のパラメータ
        ## 学習初期は 1.0に近い ＝ 探索行動(ランダムなブロック配置）が多い
        ## 学習終盤は final_epsilon に近い ＝ 活用行動(学習した結果に基づいてブロックを配置）が多い
        ## epsilon の減衰量は (final_epsilon - initial_epsilon /(num_decay_epochs - epoch) で導出
        self.initial_epsilon = cfg["train"]["initial_epsilon"]
        self.final_epsilon = cfg["train"]["final_epsilon"]
        if not isinstance(self.final_epsilon,float):
            self.final_epsilon = float(self.final_epsilon)
        self.num_decay_epochs = cfg["train"]["num_decay_epochs"]
        
        ####################################
        # 強化学習の精度を上げるための手法 
        ####################################
        # Reward Clipping
        ## 報酬を 1 で正規化、ただし消去報酬のみ...Q値の急激な変動抑制
        self.reward_clipping = cfg["train"]["reward_clipping"]
        if self.reward_clipping:
            self.norm_num =max(max(self.reward_list),abs(self.penalty))            
            self.reward_list =[r/self.norm_num for r in self.reward_list]
            self.penalty /= self.norm_num
            self.penalty = min(cfg["train"]["max_penalty"],self.penalty)
        
        ########
        # Double DQN (DDQN)
        ## 探索（行動）するネットワークと行動を評価するネットワークを分けることで学習を安定させる
        self.double_dqn = cfg["train"]["double_dqn"]
        if self.double_dqn:
            self.target_net = True

        ########
        # Target Net
        ## DDQN 使用時は自動でON(DDQN が優先される)
        ## 評価するネットワークを固定にすることで学習を安定させる
        self.target_net = cfg["train"]["target_net"]
        if self.target_net:
            print("set target network...")
            self.target_model = copy.deepcopy(self.model)
            self.target_copy_intarval = cfg["train"]["target_copy_intarval"]

        ########
        # Prioritized Experience Replay(PER)
        ## 学習時に誤差が大きいデータを優先してリプレイメモリから取り出すことで学習速度を早める
        ## PERがOFFであれば、ランダムにデータを取り出す
        self.prioritized_replay = cfg["train"]["prioritized_replay"]
        if self.prioritized_replay:
            from machine_learning.qlearning import PRIORITIZED_EXPERIENCE_REPLAY as PER
            self.PER = PER(self.replay_memory_size,gamma=self.gamma)
            
        ########
        # Multi Step learning 
        ## t手目の報酬にt+n 手目の報酬を加算する = 将来の報酬も含める
        ## テトリスではあまり効果がないかも。。
        self.multi_step_learning = cfg["train"]["multi_step_learning"]
        if self.multi_step_learning:
            from machine_learning.qlearning import Multi_Step_Learning as MSL
            self.multi_step_num = cfg["train"]["multi_step_num"]
            self.MSL = MSL(step_num=self.multi_step_num,gamma=self.gamma)

    ####################################
    # リセット時にスコア計算し episode memory に penalty 追加
    # 経験学習のために episode_memory を replay_memory 追加
    ####################################
    def stack_replay_memory(self):
        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2":
            # スコアにペナルティを加算する
            self.score += self.score_list[5]

            #[next_state, reward, next2_state, done]
            self.episode_memory[-1][1] += self.penalty
            self.episode_memory[-1][3] = True  #store False to done lists.
            self.epoch_reward += self.penalty
            #
            if self.multi_step_learning:
                self.episode_memory = self.MSL.arrange(self.episode_memory)

            # 経験学習のために episode_memory を replay_memory 追加
            self.replay_memory.extend(self.episode_memory)
            # 容量超えたら削除
            self.episode_memory = deque(maxlen=self.max_episode_size)
        else:
            pass

    ####################################
    # Game の Reset の実施 (Game Over後)
    # nextMove["option"]["reset_callback_function_addr"] へ設定
    ####################################
    def update(self):
        ########################
        # 学習の場合
        ################
        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2":
            # リセット時にスコア計算し episode memory に penalty 追加
            # replay_memory に episode memory 追加
            self.stack_replay_memory()

            # リプレイメモリが1/10たまっていないなら、
            if len(self.replay_memory) < self.replay_memory_size / 10:
                print("================pass================")
                print("iter: {} ,meory: {}/{} , score: {}, clear line: {}, block: {} ".format(self.iter,
                len(self.replay_memory),self.replay_memory_size / 10,self.score,self.cleared_lines
                ,self.tetrominoes ))
            # リプレイメモリが1/10たまっていたら
            else:
                print("================update================")
                self.epoch += 1
                
                # 優先順位つき経験学習有効なら
                if self.prioritized_replay:
                    # replay batch index 指定
                    batch,replay_batch_index = self.PER.sampling(self.replay_memory,self.batch_size)
                # そうでないなら
                else:
                    # batch 確率的勾配降下法における、全パラメータのうちランダム抽出して勾配を求めるパラメータの数 batch_size など
                    batch = sample(self.replay_memory, min(len(self.replay_memory), self.batch_size))
                    

                # batch から各情報を引き出す
                # (episode memory の並び)
                state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
                state_batch = torch.stack(tuple(state for state in state_batch))
                reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
                next_state_batch = torch.stack(tuple(state for state in next_state_batch))

                done_batch = torch.from_numpy(np.array(done_batch)[:, None])

                ###########################
                # 順伝搬し Q 値を取得 (model の __call__ ≒ forward)
                ###########################
                #max_next_state_batch = torch.stack(tuple(state for state in max_next_state_batch))
                q_values = self.model(state_batch)
                
                ###################
                # Traget net 使う場合
                if self.target_net:
                    if self.epoch %self.target_copy_intarval==0 and self.epoch>0:
                        print("target_net update...")
                        self.target_model = torch.load(self.best_weight)
                        #self.target_model = copy.copy(self.model)
                    
                    # 推論モードに変更
                    self.target_model.eval()
                    # テンソルの勾配の計算を不可とする
                    with torch.no_grad():
                        # 次の次の状態 batch から
                        # 確率的勾配降下法における batch から "ターゲット" モデルでの q 値を求める
                        next_prediction_batch = self.target_model(next_state_batch)
                else:
                    # 推論モードに変更
                    self.model.eval()
                    # テンソルの勾配の計算を不可とする
                    with torch.no_grad():
                        # 確率的勾配降下法における batch を順伝搬し Q 値を取得 (model の __call__ ≒ forward)
                        next_prediction_batch = self.model(next_state_batch)

                ####################
                # 学習モードに変更
                ##########
                self.model.train()
                
                ##########################
                # Multi Step lerning の場合 (将来の報酬を加算する)
                if self.multi_step_learning:
                    print("multi step learning update")
                    y_batch = self.MSL.get_y_batch(done_batch,reward_batch, next_prediction_batch)              
                
                # Multi Step lerning でない場合
                else:
                    # done_batch, reward_bach, next_prediction_batch(Target net など比較対象 batch)
                    # をそれぞれとりだし done が True なら reward, False (Gameover なら reward + gammma * prediction Q値)
                    # を y_batchとする (gamma は割引率)
                    y_batch = torch.cat(
                        tuple(reward if done[0] else reward + self.gamma * prediction for done ,reward, prediction in
                            zip(done_batch,reward_batch, next_prediction_batch)))[:, None]
                
                # 最適化対象のすべてのテンソルの勾配を 0 にする (逆伝搬backward 前に必須)
                self.optimizer.zero_grad()
                #########################
                ## 学習実施 - 逆伝搬
                #########################
                # 優先順位つき経験学習の場合
                if self.prioritized_replay:
                    # 優先度の更新と重みづけ取得
                    # 次の状態のbatch index
                    # 次の状態のbatch 報酬
                    # 次の状態のbatch の Q 値
                    # 次の次の状態のbatch の Q 値 (Target model 有効の場合 Target model 換算)
                    loss_weights = self.PER.update_priority(replay_batch_index,reward_batch,q_values,next_prediction_batch)
                    # 誤差関数と重みづけ計算 (q_values が現状 モデル結果, y_batch が比較対象[Target net])
                    loss = (loss_weights *self.criterion(q_values, y_batch)).mean()
                    # 逆伝搬-勾配計算
                    loss.backward()
                else:
                    loss = self.criterion(q_values, y_batch)
                    # 逆伝搬-勾配計算
                    loss.backward()
                
                # weight を更新
                self.optimizer.step()
                
                # SGD の場合
                if self.scheduler!=None:
                    # 学習率更新
                    self.scheduler.step()

                ###################################
                # 結果の出力
                log = "Epoch: {} / {}, Score: {},  block: {},  Reward: {:.1f} Cleared lines: {}".format(
                    self.epoch,
                    self.num_epochs,
                    self.score,
                    self.tetrominoes,
                    self.epoch_reward,
                    self.cleared_lines
                    )
                print(log)
                with open(self.log,"a") as f:
                    print(log, file=f)
                with open(self.log_score,"a") as f:
                    print(self.score, file=f)

                with open(self.log_reward,"a") as f:
                    print(self.epoch_reward, file=f)
                
                # TensorBoard への出力
                self.writer.add_scalar('Train/Score', self.score, self.epoch - 1) 
                self.writer.add_scalar('Train/Reward', self.epoch_reward, self.epoch - 1)   
                self.writer.add_scalar('Train/block', self.tetrominoes, self.epoch - 1)  
                self.writer.add_scalar('Train/clear lines', self.cleared_lines, self.epoch - 1) 

            ###################################
            # EPOCH 数が規定数を超えたら
            if self.epoch > self.num_epochs:
                # ログ出力
                with open(self.log,"a") as f:
                    print("finish..", file=f)
                if os.path.exists(self.latest_dir):
                    shutil.rmtree(self.latest_dir)
                os.makedirs(self.latest_dir,exist_ok=True)
                shutil.copyfile(self.best_weight,self.latest_dir+"/best_weight.pt")
                for file in glob.glob(self.output_dir+"/*.txt"):
                    shutil.copyfile(file,self.latest_dir+"/"+os.path.basename(file))
                for file in glob.glob(self.output_dir+"/*.yaml"):
                    shutil.copyfile(file,self.latest_dir+"/"+os.path.basename(file))
                with open(self.latest_dir+"/copy_base.txt","w") as f:
                    print(self.best_weight, file=f)
                    
                ####################
                # 終了
                exit() 

        ########################
        # 推論の場合
        ################
        else:
            self.epoch += 1
            log = "Epoch: {} / {}, Score: {},  block: {}, Reward: {:.1f} Cleared lines: {}".format(
            self.epoch,
            self.num_epochs,
            self.score,
            self.tetrominoes,
            self.epoch_reward,
            self.cleared_lines
            )
        ###################################
        # ゲームパラメータ初期化
        self.reset_state()

    ####################################
    #累積値の初期化 (Game Over 後)
    ####################################
    def reset_state(self):
        ## 学習の場合
        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2": 
            if self.score > self.max_score:
                ## 最高点を保存
                torch.save(self.model, "{}/tetris_epoch{}_score{}.pt".format(self.weight_dir,self.epoch,self.score))
                self.max_score  =  self.score
                torch.save(self.model,self.best_weight)
        
        # 初期化
        self.state = self.initial_state
        self.score = 0
        self.cleared_lines = 0
        self.epoch_reward = 0
        self.tetrominoes = 0
            
    ####################################
    #削除されるLineを数える
    ####################################
    def check_cleared_rows(self,board):
        board_new = np.copy(board)
        lines = 0
        empty_line = np.array([0 for i in range(self.width)])
        for y in range(self.height - 1, -1, -1):
            blockCount  = np.sum(board[y])
            if blockCount == self.width:
                lines += 1
                board_new = np.delete(board_new,y,0)
                board_new = np.vstack([empty_line,board_new ])
        return lines,board_new

    ####################################
    ## でこぼこ度, 高さ合計, 高さ最大, 高さ最小を求める
    ####################################
    def get_bumpiness_and_height(self,board):
        # ボード上で 0 でないもの(テトリミノのあるところ)を抽出
        # (1,2,3,4,5,6,7) を ブロックあり True, なし False に変更
        mask = board != 0
        # 列方向 何かブロックががあれば、そのindexを返す
        # なければ画面ボード縦サイズを返す
        # 上記を 画面ボードの列に対して実施したの配列(長さ width)を返す
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)

        # 上からの距離なので反転 (配列)
        heights = self.height - invert_heights
        # 高さの合計をとる (返り値用)
        total_height = np.sum(heights)
        # 最も高いところをとる (返り値用)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    ####################################
    ## 穴の数, 穴の上積み上げ Penalty, 最も高い穴の位置を求める
    # board: 2次元画面ボード
    ####################################
    def get_holes(self, board):
        num_holes = 0
        # 行ごとに0の数を確認
        for i in range(self.width):
            col = board[:,i]
            row = 0
            # 行ごとに最大の高さを取得
            while row < self.height and col[row] == 0:
                row += 1
            # 0 の個数を数え上げる
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    ####################################
    # 現状状態の各種パラメータ取得 (MLP
    ####################################
    def get_state_properties(self, board):
        # 削除したライン数と削除後の盤面を取得
        lines_cleared, board = self.check_cleared_rows(board)
        # 穴の数を取得
        holes = self.get_holes(board)
        # でこぼこの数と高さの一覧を取得
        bumpiness, height = self.get_bumpiness_and_height(board)
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    ####################################
    # 現状状態の各種パラメータ取得
    ####################################
    def get_state_properties_v2(self, board):
        # 削除したライン数と削除後の盤面を取得
        lines_cleared, board = self.check_cleared_rows(board)
        # 穴の数を取得
        holes = self.get_holes(board)
        # でこぼこの数と高さの一覧を取得
        bumpiness, height = self.get_bumpiness_and_height(board)
        # 最大の高さを取得
        max_row = self.get_max_height(board)
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height,max_row])

    ####################################
    # 最大の高さを取得
    ####################################
    def get_max_height(self, board):
        # X 軸のセルを足し算する
        sum_ = np.sum(board,axis=1)
        row = 0
        # X 軸の合計が0になる Y 軸を探す
        while row < self.height and sum_[row] ==0:
            row += 1
        return self.height - row

    ####################################
    #次の状態リストを取得(2次元用) DQN .... 画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
    #  get_next_func でよびだされる
    # curr_backboard 現画面
    # piece_id テトリミノ I L J T O S Z
    # currentshape_class = status["field_info"]["backboard"]
    #################################### 
    def get_next_states_v2(self,curr_backboard,piece_id,CurrentShape_class):
        states = {}
        # テトリミノごとに回転数をふりわけ
        if piece_id == 5:  # O piece
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7:
            num_rotations = 2
        else:
            num_rotations = 4

        # テトリミノ回転方向ごとに一覧追加
        for direction0 in range(num_rotations):
            x0Min, x0Max = self.getSearchXRange(CurrentShape_class, direction0)
            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                # 画面ボードデータをコピーして指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
                board = self.getBoard(curr_backboard, CurrentShape_class, direction0, x0)
                # ボードを２次元化
                reshape_backboard = self.get_reshape_backboard(board)
                # numpy to tensor (配列を1次元追加)
                reshape_backboard = torch.from_numpy(reshape_backboard[np.newaxis,:,:]).float()
                states[(x0, direction0)] = reshape_backboard
        return states

    ####################################
    #次の状態を取得(1次元用) MLP  .... 画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
    #  get_next_func でよびだされる
    ####################################
    def get_next_states(self,curr_backboard,piece_id,CurrentShape_class):
        states = {}
        # テトリミノごとに回転数をふりわけ
        if piece_id == 5:  # O piece
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7:
            num_rotations = 2
        else:
            num_rotations = 4

        # テトリミノ回転方向ごとに一覧追加
        for direction0 in range(num_rotations):
            x0Min, x0Max = self.getSearchXRange(CurrentShape_class, direction0)
            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                # 画面ボードデータをコピーして指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
                board = self.getBoard(curr_backboard, CurrentShape_class, direction0, x0)
                # ボードを２次元化
                board = self.get_reshape_backboard(board)
                states[(x0, direction0)] = self.get_state_properties(board)
        return states

    ####################################
    #ボードを２次元化
    ####################################
    def get_reshape_backboard(self,board):
        board = np.array(board)
        # 高さ, 幅で reshape
        reshape_board = board.reshape(self.height,self.width)
        # 1, 0 に変更
        reshape_board = np.where(reshape_board>0,1,0)
        return reshape_board

    ####################################
    #報酬を計算(2次元用) 
    #reward_func から呼び出される
    #################################### 
    def step_v2(self, curr_backboard,action,curr_shape_class):
        x0, direction0 = action
        # 画面ボードデータをコピーして指定座標にテトリミノを配置し落下させた画面ボードを返す
        board = self.getBoard(curr_backboard, curr_shape_class, direction0, x0)
        #ボードを２次元化
        board = self.get_reshape_backboard(board)
        # でこぼこ度, 高さ合計を求める
        bampiness,height = self.get_bumpiness_and_height(board)
        # 高さ最大を求める
        max_height = self.get_max_height(board)
        
        ## 穴の数を求める
        hole_num = self.get_holes(board)
        # 削除したライン数と削除後の盤面を取得
        lines_cleared, board = self.check_cleared_rows(board)
        # 削除したライン数から報酬値を取得
        reward = self.reward_list[lines_cleared]
        # でこぼこの数だけペナルティ
        reward -= self.reward_weight[0] *bampiness 
        ## 最大の高さだけペナルティ
        reward -= self.reward_weight[1] * max(0,max_height)
        # 穴の数だけペナルティ
        reward -= self.reward_weight[2] * hole_num
        
        # 報酬値に加算
        self.epoch_reward += reward
        # トータルスコアに加算
        self.score += self.score_list[lines_cleared]
        # 消した数に加算
        self.cleared_lines += lines_cleared
        # 落としたブロック数を加算
        self.tetrominoes += 1
        return reward

    ####################################
    #報酬を計算(1次元用) 
    #reward_func から呼び出される
    ####################################
    def step(self, curr_backboard,action,curr_shape_class):
        x0, direction0 = action
        # 画面ボードデータをコピーして指定座標にテトリミノを配置し落下させた画面ボードを返す
        board = self.getBoard(curr_backboard, curr_shape_class, direction0, x0)
        #ボードを２次元化
        board = self.get_reshape_backboard(board)
        # 削除したライン数と削除後の盤面を取得
        lines_cleared, board = self.check_cleared_rows(board)
        # 削除したライン数から報酬値を取得
        reward = self.reward_list[lines_cleared]
        # 報酬値に加算
        self.epoch_reward += reward
        # トータルスコアに加算
        self.score += self.score_list[lines_cleared]
        # 消した数に加算
        self.cleared_lines += lines_cleared
        # 落としたブロック数を加算
        self.tetrominoes += 1
        return reward

    ####################################
    # 次の動作取得: ゲームコントローラから毎回呼ばれる
    ####################################
    def GetNextMove(self, nextMove, GameStatus,yaml_file=None,weight=None):

        t1 = datetime.now()
        # RESET 関数設定 callback function 代入 (Game Over 時)
        nextMove["option"]["reset_callback_function_addr"] = self.update
        
        # mode の取得
        self.mode = GameStatus["judge_info"]["mode"]
        
        ################
        ## 初期パラメータない場合は初期パラメータ読み込み
        if self.init_train_parameter_flag == False:
            self.init_train_parameter_flag = True
            self.set_parameter(yaml_file=yaml_file,predict_weight=weight)        
        self.ind =GameStatus["block_info"]["currentShape"]["index"]
        curr_backboard = GameStatus["field_info"]["backboard"]

        ##################
        # default board definition
        # self.width, self.height と重複
        self.board_data_width = GameStatus["field_info"]["width"]
        self.board_data_height = GameStatus["field_info"]["height"]

        curr_shape_class = GameStatus["block_info"]["currentShape"]["class"]
        next_shape_class= GameStatus["block_info"]["nextShape"]["class"]

        ##################
        # next shape info
        self.ShapeNone_index = GameStatus["debug_info"]["shape_info"]["shapeNone"]["index"]
        curr_piece_id =GameStatus["block_info"]["currentShape"]["index"]
        next_piece_id =GameStatus["block_info"]["nextShape"]["index"]
        reshape_backboard = self.get_reshape_backboard(curr_backboard)
               
        ###################
        #画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
        # next_steps
        #    Key = Tuple (テトリミノ画面ボードX座標, テトリミノ回転状態)
        #                 テトリミノ Move Down 降下 数, テトリミノ追加移動X座標, テトリミノ追加回転)
        #    Value = 画面ボード状態
        next_steps =self.get_next_func(curr_backboard,curr_piece_id,curr_shape_class)

        ###############################################
        # 学習の場合
        ###############################################
        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2":
            ###################
            # epsilon-greedy法
            ## 学習初期はランダムに行動(探索)し、終盤は学習した
            ## ニューラルネットワークに従って行動(活用)
            
            # epsilon = 学習結果から乱数で変更する割合対象
            # num_decay_epochs より前までは比例で初期 epsilon から減らしていく
            epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.epoch, 0) * (
                    self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
            # 乱数を取得
            u = random()
            
            # 乱数がepsilonより小さければ,True(ランダム行動)
            random_action = u <= epsilon
    
            # 次の状態一覧の action と states で配列化
            # next_actions  = Tuple (テトリミノ画面ボードX座標, テトリミノ回転状態)　一覧
            # next_states = 画面ボード状態 一覧
            next_actions, next_states = zip(*next_steps.items())
            # next_states (画面ボード状態 一覧) のテンソルを連結 (画面ボード状態のlist の最初の要素に状態が追加された)
            next_states = torch.stack(next_states)

            # 学習モードに変更
            self.model.train()
            # テンソルの勾配の計算を不可とする(Tensor.backward() を呼び出さないことが確実な場合)
            with torch.no_grad():
                # 順伝搬し Q 値を取得 (model の __call__ ≒ forward)
                predictions = self.model(next_states)[:, 0]
            
            # 乱数が epsilon より小さい場合はランダムな行動
            if random_action:
                # index を乱数とする
                index = randint(0, len(next_steps) - 1)
            # 乱数が epsilon より大きい場合は推論した結果を活用
            else:
                # index を推論の最大値とする
                index = torch.argmax(predictions).item()

            # 次の action states を上記の index 元に決定
            next_state = next_states[index, :]

            # index にて次の action の決定 
            # action の list
            # 0: 2番目 X軸移動
            # 1: 1番目 テトリミノ回転
            # 2: 3番目 Y軸降下 (-1: で Drop)
            # 3: 4番目 テトリミノ回転 (Next Turn)
            # 4: 5番目 X軸移動 (Next Turn)
            action = next_actions[index]
            
            # step, step_v2 により報酬計算
            reward = self.reward_func(curr_backboard,action,curr_shape_class)
            
            done = False #game over flag
            
            ################################
            # Double DQN 有効時
            #======predict max_a Q(s_(t+1),a)======
            if self.double_dqn:
                # 画面ボードデータをコピーして 指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
                next_backboard  = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0])
                
                #画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
                next2_steps =self.get_next_func(next_backboard,next_piece_id,next_shape_class)
                # 次の状態一覧の action と states で配列化
                next2_actions, next2_states = zip(*next2_steps.items())
                # next_states のテンソルを連結
                next2_states = torch.stack(next2_states)

                # 学習モードに変更
                self.model.train()
                # テンソルの勾配の計算を不可とする
                with torch.no_grad():
                    # 順伝搬し Q 値を取得 (model の __call__ ≒ forward)
                    next_predictions = self.model(next2_states)[:, 0]
                # 次の index を推論の最大値とする
                next_index = torch.argmax(next_predictions).item()
                # 次の状態を index で指定し取得
                next2_state = next2_states[next_index, :]

            ################################
            # Target Next 有効時
            elif self.target_net:
                # 画面ボードデータをコピーして 指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
                next_backboard  = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0])
                #画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
                next2_steps =self.get_next_func(next_backboard,next_piece_id,next_shape_class)
                # 次の状態一覧の action と states で配列化
                next2_actions, next2_states = zip(*next2_steps.items())
                # next_states のテンソルを連結
                next2_states = torch.stack(next2_states)
                # 学習モードに変更
                self.target_model.train()
                # テンソルの勾配の計算を不可とする
                with torch.no_grad():
                    # 順伝搬し Q 値を取得 (model の __call__ ≒ forward)
                    next_predictions = self.target_model(next2_states)[:, 0]
                # 次の index を推論の最大値とする
                next_index = torch.argmax(next_predictions).item()
                # 次の状態を index で指定し取得
                next2_state = next2_states[next_index, :]

            ################################
            # DDQN 、Targetを使用しない場合
            else:
                # 画面ボードデータをコピーして 指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
                next_backboard  = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0])
                #画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
                next2_steps =self.get_next_func(next_backboard,next_piece_id,next_shape_class)
                # 次の状態一覧の action と states で配列化
                next2_actions, next2_states = zip(*next2_steps.items())
                # next_states のテンソルを連結
                next2_states = torch.stack(next2_states)
                # 学習モードに変更
                self.model.train()
                # テンソルの勾配の計算を不可とする
                with torch.no_grad():
                    # 順伝搬し Q 値を取得 (model の __call__ ≒ forward)
                    next_predictions = self.model(next2_states)[:, 0]
                # 乱数を取得
                u = random()
                # 乱数が epsilon より小さい場合は
                random_action = u <= epsilon
                if random_action:
                    # index を乱数指定
                    next_index = randint(0, len(next2_steps) - 1)
                else:
                    # 次の index を推論の最大値とする
                    next_index = torch.argmax(next_predictions).item()
                # 次の状態を index により指定
                next2_state = next2_states[next_index, :]
                
            ##########################
            # Episode Memory に
            # next_state  次の候補第1位手
            # reward 報酬
            # next2_state 比較対象のモデルによる候補手 (Target net など)
            # done Game Over flag
            self.episode_memory.append([next_state, reward, next2_state,done])
            if self.prioritized_replay:
                # キューにリプレイ用の情報を格納していく
                self.PER.store()
            
            ########################
            ## 学習時 次の動作指定
            ################
            # テトリミノ回転
            nextMove["strategy"]["direction"] = action[1]
            # 横方向
            nextMove["strategy"]["x"] = action[0]
            # Drop Down 落下
            nextMove["strategy"]["y_operation"] = 1
            # Move Down 降下数
            nextMove["strategy"]["y_moveblocknum"] = 1
            
            # 規定値よりブロック数が多ければ、強制的にリセットする
            if self.tetrominoes > self.max_tetrominoes:
                nextMove["option"]["force_reset_field"] = True
            # STATE = next_state 代入
            self.state = next_state

        ###############################################
        # 推論 の場合
        ###############################################
        elif self.mode == "predict" or self.mode == "predict_sample" or self.mode == "predict_sample2":
            #推論モードに切り替え
            self.model.eval()
            ### 画面ボードの次の状態一覧を action と states にわけ、states を連結
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            ## 順伝搬し Q 値を取得 (model の __call__ ≒ forward)
            predictions = self.model(next_states)[:, 0]
            ## 最大値の index 取得
            index = torch.argmax(predictions).item()
            action = next_actions[index]
            
            ########################
            ## 推論時 次の動作指定
            ################
            # テトリミノ回転
            nextMove["strategy"]["direction"] = action[1]
            # 横方向
            nextMove["strategy"]["x"] = action[0]
            # Drop Down 落下
            nextMove["strategy"]["y_operation"] = 1
            # Move Down 降下数
            nextMove["strategy"]["y_moveblocknum"] = 1
        return nextMove

    ####################################
    # テトリミノが配置できる左端と右端の座標を返す
    # self,
    # Shape_class: 現在と予告テトリミノの配列
    # direction: 現在のテトリミノ方向
    ####################################
    def getSearchXRange(self, Shape_class, direction):
        #
        # get x range from shape direction.
        #
        # テトリミノが原点から x 両方向に最大何マス占有するのか取得
        minX, maxX, _, _ = Shape_class.getBoundingOffsets(direction) # get shape x offsets[minX,maxX] as relative value.
        # 左方向のサイズ分
        xMin = -1 * minX
        # 右方向のサイズ分（画面サイズからひく）
        xMax = self.board_data_width - maxX
        return xMin, xMax

    ####################################
    # direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置した場合の2次元座標配列を返す
    ####################################
    def getShapeCoordArray(self, Shape_class, direction, x, y):
        #
        # get coordinate array by given shape.
        #
        # direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置した場合の2次元座標配列を返す
        coordArray = Shape_class.getCoords(direction, x, y) # get array from shape direction, x, y.
        return coordArray

    ####################################
    # 画面ボードデータをコピーして指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
    # board_backboard: 現状画面ボード
    # Shape_class: テトリミノ現/予告リスト
    # direction: テトリミノ回転方向
    # x: テトリミノx座標
    ####################################
    def getBoard(self, board_backboard, Shape_class, direction, x):
        # 
        # get new board.
        #
        # copy backboard data to make new board.
        # if not, original backboard data will be updated later.
        board = copy.deepcopy(board_backboard)
        # 指定座標から落下させたところにテトリミノを固定しその画面ボードを返す
        _board = self.dropDown(board, Shape_class, direction, x)
        return _board

    ####################################
    # 指定座標から落下させたところにテトリミノを固定しその画面ボードを返す
    # board: 現状画面ボード
    # Shape_class: テトリミノ現/予告リスト
    # direction: テトリミノ回転方向
    # x: テトリミノx座標
    ####################################
    def dropDown(self, board, Shape_class, direction, x):
        # 
        # internal function of getBoard.
        # -- drop down the shape on the board.
        # 
        # 画面ボード下限座標として dy 設定
        dy = self.board_data_height - 1
        # direction (回転状態)のテトリミノ2次元座標配列を取得し、それをx,yに配置した場合の座標配列を返す
        coordArray = self.getShapeCoordArray(Shape_class, direction, x, 0)
        # update dy
        # テトリミノ座標配列ごとに...
        for _x, _y in coordArray:
            _yy = 0
            # _yy を一つずつ落とすことによりブロックの落下下限を確認
            # _yy+テトリミノ座標y が 画面下限より上　かつ　(_yy +テトリミノ座標yが画面上限より上 または テトリミノ座標_x,_yy+テトリミノ座標_yのブロックがない)
            while _yy + _y < self.board_data_height and (_yy + _y < 0 or board[(_y + _yy) * self.board_data_width + _x] == self.ShapeNone_index):
                #_yy を足していく(下げていく)
                _yy += 1
            _yy -= 1
            # 下限座標 dy /今までの下限より小さい(高い)なら __yy を落下下限として設定
            if _yy < dy:
                dy = _yy
        # get new board
        _board = self.dropDownWithDy(board, Shape_class, direction, x, dy)
        return _board

    ####################################
    # 指定位置にテトリミノを固定する
    # board: 現状画面ボード
    # Shape_class: テトリミノ現/予告リスト
    # direction: テトリミノ回転方向
    # x: テトリミノx座標
    ####################################
    def dropDownWithDy(self, board, Shape_class, direction, x, dy):
        #
        # internal function of dropDown.
        # board コピー
        _board = board
        coordArray = self.getShapeCoordArray(Shape_class, direction, x, 0)
        # テトリミノ座標配列を順に進める
        for _x, _y in coordArray:
            #x, dy の 画面ボードにブロックを配置して、その画面ボードデータを返す
            _board[(_y + dy) * self.board_data_width + _x] = Shape_class.shape
        return _board
BLOCK_CONTROLLER_TRAIN = Block_Controller()
