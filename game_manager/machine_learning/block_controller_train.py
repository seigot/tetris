#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime
import pprint
import random
import copy
import torch
import torch.nn as nn
import sys
sys.path.append("game_manager/machine_learning/")
#import omegaconf
#from hydra import compose, initialize
import os
from tensorboardX import SummaryWriter
from collections import deque
from random import random, sample,randint
import shutil
import glob 
import numpy as np
import yaml
#import subprocess

###################################################
###################################################
# ブロック操作クラス
###################################################
###################################################
class Block_Controller(object):

    ####################################
    # 起動時初期化
    # init parameter
    board_backboard = 0
    board_data_width = 0
    board_data_height = 0
    ShapeNone_index = 0
    CurrentShape_class = 0
    NextShape_class = 0

    ## 第2weight
    # 有効かどうか
    weight2_available = False
    # ゲーム途中の切り替えフラグ
    weight2_enable = False
    predict_weight2_enable_index = 0
    predict_weight2_disable_index = 0

    # Debug 出力
    debug_flag_shift_rotation = 0
    debug_flag_shift_rotation_success = 0
    debug_flag_try_move = 0
    debug_flag_drop_down = 0
    debug_flag_move_down = 0

    ####################################
    # 起動時初期化
    ####################################
    def __init__(self):
        # init parameter
        self.mode = None
        # train
        self.init_train_parameter_flag = False
        # predict
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
        self.result_warehouse = "outputs/"
        self.latest_dir = self.result_warehouse+"/latest"
        predict_weight2 = None

        ########
        ## Config Yaml 読み込み
        if yaml_file is None:
            raise Exception('Please input train_yaml file.')
        elif not os.path.exists(yaml_file):
            raise Exception('The yaml file {} is not existed.'.format(yaml_file))
        cfg = self.yaml_read(yaml_file)

        ########
        ## 学習の場合
        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2":
            # ouput dir として日付ディレクトリ作成
            dt = datetime.now()
            self.output_dir = self.result_warehouse+ dt.strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs(self.output_dir,exist_ok=True)
            
            # weight_dir として output_dir 下に trained model フォルダを output_dir 傘下に作る
            self.weight_dir = self.output_dir+"/trained_model/"
            self.best_weight = self.weight_dir + "best_weight.pt"
            os.makedirs(self.weight_dir,exist_ok=True)
        ########
        ## 推論の場合
        else:
            ## Config Yaml で指定の場合
            predict_weight_cfg = True
            if ('predict_weight' in cfg["common"]) \
                    and (predict_weight == "outputs/latest/best_weight.pt"):
                predict_weight = cfg["common"]["predict_weight"]
                predict_weight_cfg = True
            else:
                predict_weight_cfg = False

            dirname = os.path.dirname(predict_weight)
            self.output_dir = dirname + "/predict/"
            os.makedirs(self.output_dir,exist_ok=True)

            ## 第2 model
            self.weight2_available = False
            self.weight2_enable = False
            # config yaml の weight2_available が True, かつ predict_weight2 がありかつ predict_weight が指定でない場合
            if ('weight2_available' in cfg["common"]) \
                    and cfg["common"]["weight2_available"] \
                    and cfg["common"]["predict_weight2"] != None \
                    and predict_weight_cfg:
                self.weight2_available = True
                predict_weight2 = cfg["common"]["predict_weight2"]
                self.predict_weight2_enable_index = cfg["common"]["predict_weight2_enable_index"]
                self.predict_weight2_disable_index = cfg["common"]["predict_weight2_disable_index"]

        ####################
        # default.yaml を output_dir にコピーしておく
        #subprocess.run("cp config/default.yaml %s/"%(self.output_dir), shell=True)
        shutil.copy2(yaml_file, self.output_dir)

        # Tensorboard 出力フォルダ設定
        self.writer = SummaryWriter(self.output_dir+"/"+cfg["common"]["log_path"])

        ####################
        # ログファイル設定
        ########
        # 推論の場合
        if self.mode=="predict" or self.mode=="predict_sample":
            self.log = self.output_dir+"/log_predict.txt"
            self.log_score = self.output_dir+"/score_predict.txt"
            self.log_reward = self.output_dir+"/reward_predict.txt"
        ########
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

        # Move Down 降下有効化
        if 'move_down_flag' in cfg["train"]:
            self.move_down_flag = cfg["train"]["move_down_flag"]
        else:
            self.move_down_flag = 0

        # 次のテトリミノ予測数
        if cfg["model"]["name"]=="DQN" and ('predict_next_num' in cfg["train"]):
            self.predict_next_num = cfg["train"]["predict_next_num"]
        else:
            self.predict_next_num = 0

        # 次のテトリミノ候補数
        if cfg["model"]["name"]=="DQN" and ('predict_next_steps' in cfg["train"]):
            self.predict_next_steps = cfg["train"]["predict_next_steps"]
        else:
            self.predict_next_steps = 0

        # 次のテトリミノ予測数 (学習時)
        if cfg["model"]["name"]=="DQN" and ('predict_next_num_train' in cfg["train"]):
            self.predict_next_num_train = cfg["train"]["predict_next_num_train"]
        else:
            self.predict_next_num_train = 0

        # 次のテトリミノ候補数 (学習時)
        if cfg["model"]["name"]=="DQN" and ('predict_next_steps_train' in cfg["train"]):
            self.predict_next_steps_train = cfg["train"]["predict_next_steps_train"]
        else:
            self.predict_next_steps_train = 0

        # 終了時刻表示
        if 'time_disp' in cfg["common"]:
            self.time_disp = cfg["common"]["time_disp"]
        else:
            self.time_disp = False

        ####################
        #=====Set tetris parameter=====
        # Tetris ゲーム指定
        # self.board_data_width , self.board_data_height と二重指定、統合必要
        self.height = cfg["tetris"]["board_height"]
        self.width = cfg["tetris"]["board_width"]
        
        # 最大テトリミノ
        self.max_tetrominoes = cfg["tetris"]["max_tetrominoes"]
        
        ####################
        # ニューラルネットワークの入力数
        self.state_dim = cfg["state"]["dim"]
        # 学習+推論方式
        print("model name: %s"%(cfg["model"]["name"]))

        ### config/default.yaml で選択
        ## MLP の場合
        if cfg["model"]["name"]=="MLP":
            #=====load MLP=====
            # model/deepnet.py の MLP 読み込み
            from machine_learning.model.deepqnet import MLP
            # 入力数設定して MLP モデルインスタンス作成
            self.model = MLP(self.state_dim)
            # 初期状態規定
            self.initial_state = torch.FloatTensor([0 for i in range(self.state_dim)])
            #各関数規定
            self.get_next_func = self.get_next_states
            self.reward_func = self.step
            # 報酬関連規定
            self.reward_weight = cfg["train"]["reward_weight"]
            # 穴の上の積み上げペナルティ
            self.hole_top_limit = 1
            # 穴の上の積み上げペナルティ下限絶対高さ
            self.hole_top_limit_height = -1

        # DQN の場合
        elif cfg["model"]["name"]=="DQN":
            #=====load Deep Q Network=====
            from machine_learning.model.deepqnet import DeepQNetwork
            # DQN モデルインスタンス作成
            self.model = DeepQNetwork()
            if self.weight2_available:
                self.model2 = DeepQNetwork()

            # 初期状態規定
            self.initial_state = torch.FloatTensor([[[0 for i in range(10)] for j in range(22)]])
            #各関数規定
            self.get_next_func = self.get_next_states_v2
            self.reward_func = self.step_v2
            # 報酬関連規定
            self.reward_weight = cfg["train"]["reward_weight"]
            
            if 'tetris_fill_reward' in cfg["train"]:
                self.tetris_fill_reward = cfg["train"]["tetris_fill_reward"]
            else:
                self.tetris_fill_reward = 0
            print("tetris_fill_reward:", self.tetris_fill_reward)

            if 'tetris_fill_height' in cfg["train"]:
                self.tetris_fill_height = cfg["train"]["tetris_fill_height"]
            else:
                self.tetris_fill_height = 0
            print("tetris_fill_height:", self.tetris_fill_height)

            if 'height_line_reward' in cfg["train"]:
                self.height_line_reward = cfg["train"]["height_line_reward"]
            else:
                self.height_line_reward = 0
            print("height_line_reward:", self.height_line_reward)
    
            if 'hole_top_limit_reward' in cfg["train"]:
                self.hole_top_limit_reward = cfg["train"]["hole_top_limit_reward"]
            else:
                self.hole_top_limit_reward = 0
            print("hole_top_limit_reward:", self.hole_top_limit_reward)
    
            # 穴の上の積み上げペナルティ
            if 'hole_top_limit' in cfg["train"]:
                self.hole_top_limit = cfg["train"]["hole_top_limit"]
            else:
                self.hole_top_limit = 1
            print("hole_top_limit:", self.hole_top_limit)
    
            # 穴の上の積み上げペナルティ下限絶対高さ
            if 'hole_top_limit_height' in cfg["train"]:
                self.hole_top_limit_height = cfg["train"]["hole_top_limit_height"]
            else:
                self.hole_top_limit_height = -1
            print("hole_top_limit_height:", self.hole_top_limit_height)

            if 'left_side_height_penalty' in cfg["train"]:
                self.left_side_height_penalty = cfg["train"]["left_side_height_penalty"]
            else:
                self.left_side_height_penalty = 0
            print("left_side_height_penalty:", self.left_side_height_penalty)


        # 共通報酬関連規定
        if 'bumpiness_left_side_relax' in cfg["train"]:
            self.bumpiness_left_side_relax = cfg["train"]["bumpiness_left_side_relax"]
        else:
            self.bumpiness_left_side_relax = 0
        print("bumpiness_left_side_relax:", self.bumpiness_left_side_relax)
            
        if 'max_height_relax' in cfg["train"]:
            self.max_height_relax = cfg["train"]["max_height_relax"]
        else:
            self.max_height_relax = 0
        print("max_height_relax:", self.max_height_relax)



        ####################
        # 推論の場合 推論ウェイトを torch　で読み込み model に入れる。
        if self.mode=="predict" or self.mode=="predict_sample":
            if not predict_weight=="None":
                if os.path.exists(predict_weight):
                    print("Load {}...".format(predict_weight))
                    # 推論インスタンス作成
                    self.model = torch.load(predict_weight)
                    # インスタンスを推論モードに切り替え
                    self.model.eval()    
                else:
                    print("{} is not existed!!".format(predict_weight))
                    exit()
            else:
                print("Please set predict_weight!!")
                exit()

            ## 第2 model
            if self.weight2_available and (not predict_weight2=="None"):
                if os.path.exists(predict_weight2):
                    print("Load2 {}...".format(predict_weight2))
                    # 推論インスタンス作成
                    self.model2 = torch.load(predict_weight2)
                    # インスタンスを推論モードに切り替え
                    self.model2.eval()    
                else:
                    print("{} is not existed!!(predict 2)".format(predict_weight))
                    exit()

        ####################
        #### finetune の場合
        #(以前の学習結果を使う場合　
        elif cfg["model"]["finetune"]:
            # weight ファイル(以前の学習ファイル)を指定
            self.ft_weight = cfg["common"]["ft_weight"]
            if not self.ft_weight is None:
                ## 読み込んでインスタンス作成
                self.model = torch.load(self.ft_weight)
                ## ログへ出力
                with open(self.log,"a") as f:
                    print("Finetuning mode\nLoad {}...".format(self.ft_weight), file=f)
                
        ## GPU 使用できるときは使う
#        if torch.cuda.is_available():
#            self.model.cuda()
        
        #=====Set hyper parameter=====
        #  学習バッチサイズ(学習の分割単位, データサイズを分割している)
        self.batch_size = cfg["train"]["batch_size"]
        # lr = learning rate　学習率
        self.lr = cfg["train"]["lr"]
        # pytorch 互換性のためfloat に変換
        if not isinstance(self.lr,float):
            self.lr = float(self.lr)
        # リプレイメモリサイズ
        self.replay_memory_size = cfg["train"]["replay_memory_size"]
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        # 最大 Episode サイズ = 最大テトリミノ数
        # 1 Episode = 1 テトリミノ
        self.max_episode_size = self.max_tetrominoes
        self.episode_memory = deque(maxlen=self.max_episode_size)
        # 学習率減衰効果を出す EPOCH 数　(1 EPOCH = 1ゲーム)
        self.num_decay_epochs = cfg["train"]["num_decay_epochs"]
        # EPOCH 数
        self.num_epochs = cfg["train"]["num_epoch"]
        # epsilon: 過去の学習結果から変更する割合 initial は初期値、final は最終値
        # Fine Tuning 時は initial を小さめに
        self.initial_epsilon = cfg["train"]["initial_epsilon"]
        self.final_epsilon = cfg["train"]["final_epsilon"]
        # pytorch 互換性のためfloat に変換
        if not isinstance(self.final_epsilon,float):
            self.final_epsilon = float(self.final_epsilon)

        ## 損失関数（予測値と、実際の正解値の誤差）と勾配法(ADAM or SGD) の決定 
        #=====Set loss function and optimizer=====
        # ADAM の場合 .... 移動平均で振動を抑制するモーメンタム と 学習率を調整して振動を抑制するRMSProp を組み合わせている
        if cfg["train"]["optimizer"]=="Adam" or cfg["train"]["optimizer"]=="ADAM":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.scheduler = None
        # ADAM でない場合SGD (確率的勾配降下法、モーメンタムも STEP SIZE も学習率γもスケジューラも設定)
        else:
            # モーメンタム設定　今までの移動とこれから動くべき移動の平均をとり振動を防ぐための関数
            self.momentum =cfg["train"]["lr_momentum"] 
            # SGD に設定
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
            # 学習率更新タイミングの EPOCH 数
            self.lr_step_size = cfg["train"]["lr_step_size"]
            # 学習率γ設定　...  Step Size 進んだ EPOCH で gammma が学習率に乗算される
            self.lr_gamma = cfg["train"]["lr_gamma"]
            # 学習率スケジューラ
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size , gamma=self.lr_gamma)
        # 誤差関数 - MSELoss 平均二乗誤差
        self.criterion = nn.MSELoss()

        ####各パラメータ初期化
        ####=====Initialize parameter=====
        #1EPOCH ... 1試行
        self.epoch = 0
        self.score = 0
        self.max_score = -99999
        self.epoch_reward = 0
        self.cleared_lines = 0
        self.cleared_col = [0,0,0,0,0]
        self.iter = 0 
        # 初期ステート
        self.state = self.initial_state 
        # テトリミノ0
        self.tetrominoes = 0

        # 前のターンで Drop をスキップしていたか？ (-1: していない, それ以外: していた)
        # third_y, forth_direction, fifth_x
        self.skip_drop = [-1, -1, -1]

        # γ 割引率 = 将来の価値をどの程度下げるか
        self.gamma = cfg["train"]["gamma"]
        # 報酬を1 で正規化するかどうか、ただし消去報酬のみ　
        self.reward_clipping = cfg["train"]["reward_clipping"]

        self.score_list = cfg["tetris"]["score_list"]
        # 報酬読み込み
        self.reward_list = cfg["train"]["reward_list"]
        # Game Over 報酬 = Penalty
        self.penalty =  self.reward_list[5]

        ########
        # 報酬を 1 で正規化、ただし消去報酬のみ...Q値の急激な変動抑制
        #=====Reward clipping=====
        if self.reward_clipping:
            # 報酬リストとペナルティ(GAMEOVER 報酬)リストの絶対値の最大をとる
            self.norm_num =max(max(self.reward_list),abs(self.penalty))            
            # 最大値で割った値を改めて報酬リストとする
            self.reward_list =[r/self.norm_num for r in self.reward_list]
            # ペナルティリストも同じようにする
            self.penalty /= self.norm_num
            # max_penalty 設定と penalty 設定の小さい方を新たに ペナルティ値とする
            self.penalty = min(cfg["train"]["max_penalty"],self.penalty)

        #########
        #=====Double DQN=====
        self.double_dqn = cfg["train"]["double_dqn"]
        self.target_net = cfg["train"]["target_net"]
        if self.double_dqn:
            self.target_net = True

        #Target_net ON ならば
        if self.target_net:
            print("set target network...")
            # 機械学習モデル複製
            self.target_model = copy.deepcopy(self.model)
            self.target_copy_intarval = cfg["train"]["target_copy_intarval"]

        ########
        #=====Prioritized Experience Replay=====
        # 優先順位つき経験学習有効ならば
        self.prioritized_replay = cfg["train"]["prioritized_replay"]
        if self.prioritized_replay:
            from machine_learning.qlearning import PRIORITIZED_EXPERIENCE_REPLAY as PER
            # 優先順位つき経験学習設定
            self.PER = PER(self.replay_memory_size, gamma=self.gamma, alpha=0.7, beta=0.5)

        ########
        #=====Multi step learning=====
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

        ##############################
        ## 学習の場合
        ##############################
        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2":
            # リセット時にスコア計算し episode memory に penalty 追加
            # replay_memory に episode memory 追加
            self.stack_replay_memory()

            ##############################
            ## ログ表示
            ##############################
            # リプレイメモリが1/10たまっていないなら、
            if len(self.replay_memory) < self.replay_memory_size / 10:
                print("================pass================")
                print("iter: {} ,meory: {}/{} , score: {}, clear line: {}, block: {}, col1-4: {}/{}/{}/{} ".format(self.iter,
                len(self.replay_memory),self.replay_memory_size / 10,self.score,self.cleared_lines
                ,self.tetrominoes, self.cleared_col[1], self.cleared_col[2], self.cleared_col[3], self.cleared_col[4]))
            # リプレイメモリがいっぱいなら
            else:
                print("================update================")
                self.epoch += 1
                # 優先順位つき経験学習有効なら
                if self.prioritized_replay:
                    # replay batch index 指定
                    batch, replay_batch_index = self.PER.sampling(self.replay_memory, self.batch_size)
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
                        # self.target_copy_intarval ごとに best_weight を target に切り替え
                        self.target_model = torch.load(self.best_weight)
                        #self.target_model = copy.copy(self.model)
                    # インスタンスを推論モードに切り替え
                    self.target_model.eval()
                    #======predict Q(S_t+1 max_a Q(s_(t+1),a))======
                    # テンソルの勾配の計算を不可とする
                    with torch.no_grad():
                        # 次の次の状態 batch から
                        # 確率的勾配降下法における batch から "ターゲット" モデルでの q 値を求める
                        next_prediction_batch = self.target_model(next_state_batch)
                else:
                    # インスタンスを推論モードに切り替え
                    self.model.eval()
                    # テンソルの勾配の計算を不可とする
                    with torch.no_grad():
                        # 確率的勾配降下法における batch を順伝搬し Q 値を取得 (model の __call__ ≒ forward)
                        next_prediction_batch = self.model(next_state_batch)

                ##########################
                # モデルの学習実施
                ##########################
                self.model.train()
                
                ##########################
                # Multi Step lerning の場合
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
                    #print(loss_weights *nn.functional.mse_loss(q_values, y_batch))
                    # 誤差関数と重みづけ計算 (q_values が現状 モデル結果, y_batch が比較対象[Target net])
                    loss = (loss_weights *self.criterion(q_values, y_batch)).mean()
                    #loss = self.criterion(q_values, y_batch)
                    
                    # 逆伝搬-勾配計算
                    loss.backward()
                else:
                    loss = self.criterion(q_values, y_batch)
                    # 逆伝搬-勾配計算
                    loss.backward()
                # weight を学習率に基づき更新
                self.optimizer.step()
                # SGD の場合
                if self.scheduler!=None:
                    # 学習率更新
                    self.scheduler.step()

                ###################################
                # 結果の出力
                log = "Epoch: {} / {}, Score: {},  block: {},  Reward: {:.4f} Cleared lines: {}, col: {}/{}/{}/{} ".format(
                    self.epoch,
                    self.num_epochs,
                    self.score,
                    self.tetrominoes,
                    self.epoch_reward,
                    self.cleared_lines,
                    self.cleared_col[1],
                    self.cleared_col[2],
                    self.cleared_col[3],
                    self.cleared_col[4]
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

                self.writer.add_scalar('Train/1 line', self.cleared_col[1], self.epoch - 1) 
                self.writer.add_scalar('Train/2 line', self.cleared_col[2], self.epoch - 1) 
                self.writer.add_scalar('Train/3 line', self.cleared_col[3], self.epoch - 1) 
                self.writer.add_scalar('Train/4 line', self.cleared_col[4], self.epoch - 1) 

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

        ###################################
        # 推論の場合
        else:
            self.epoch += 1
            log = "Epoch: {} / {}, Score: {},  block: {}, Reward: {:.4f} Cleared lines: {}- {}/ {}/ {}/ {}".format(
            self.epoch,
            self.num_epochs,
            self.score,
            self.tetrominoes,
            self.epoch_reward,
            self.cleared_lines,
            self.cleared_col[1],
            self.cleared_col[2],
            self.cleared_col[3],
            self.cleared_col[4]
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
            ## 最高点,500 epoch おきに保存
            if self.score > self.max_score or self.epoch % 500 == 0:
                torch.save(self.model, "{}/tetris_epoch{}_score{}.pt".format(self.weight_dir,self.epoch,self.score))
                self.max_score  =  self.score
                torch.save(self.model,self.best_weight)
        # 初期化ステート
        self.state = self.initial_state
        self.score = 0
        self.cleared_lines = 0
        self.cleared_col = [0,0,0,0,0]
        self.epoch_reward = 0
        # テトリミノ 0 個
        self.tetrominoes = 0
        # 前のターンで Drop をスキップしていたか？ (-1: していない, それ以外: していた)
        # third_y, forth_direction, fifth_x
        self.skip_drop = [-1, -1, -1]

    ####################################
    #削除されるLineを数える
    ####################################
    def check_cleared_rows(self, reshape_board):
        board_new = np.copy(reshape_board)
        lines = 0
        empty_line = np.array([0 for i in range(self.width)])
        for y in range(self.height - 1, -1, -1):
            blockCount  = np.sum(reshape_board[y])
            if blockCount == self.width:
                lines += 1
                board_new = np.delete(board_new,y,0)
                board_new = np.vstack([empty_line,board_new ])
        return lines,board_new

    ####################################
    ## でこぼこ度, 高さ合計, 高さ最大, 高さ最小を求める
    ####################################
    def get_bumpiness_and_height(self, reshape_board):
        # ボード上で 0 でないもの(テトリミノのあるところ)を抽出
        # (0,1,2,3,4,5,6,7) を ブロックあり True, なし False に変更
        mask = reshape_board != 0
        #pprint.pprint(mask, width = 61, compact = True)

        # 列方向 何かブロックががあれば、そのindexを返す
        # なければ画面ボード縦サイズを返す
        # 上記を 画面ボードの列に対して実施したの配列(長さ width)を返す
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        # 上からの距離なので反転 (配列)
        heights = self.height - invert_heights
        # 高さの合計をとる (返り値用)
        total_height = np.sum(heights)
        # 最も高いところをとる (返り値用)
        max_height = np.max(heights)
        # 最も低いところをとる (返り値用)
        min_height = np.min(heights)

        # 右端列を削った 高さ配列
        #currs = heights[:-1]
        currs = heights[1:-1]

        # 左端列2つを削った高さ配列
        #nexts = heights[1:]
        nexts = heights[2:]

        # 差分の絶対値をとり配列にする
        diffs = np.abs(currs - nexts)
        # 左端列は self.bumpiness_left_side_relax 段差まで許容
        if heights[1] - heights[0] > self.bumpiness_left_side_relax or heights[1] - heights[0] < 0 :
            diffs = np.append(abs(heights[1] - heights[0]), diffs)

        # 差分の絶対値を合計してでこぼこ度とする
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height, max_height, min_height, heights[0]

    ####################################
    ## 穴の数, 穴の上積み上げ Penalty, 最も高い穴の位置を求める
    # reshape_board: 2次元画面ボード
    # min_height: 到達可能の最下層より1行下の穴の位置をチェック -1 で無効 hole_top_penalty 無効
    ####################################
    def get_holes(self, reshape_board, min_height):
        # 穴の数
        num_holes = 0
        # 穴の上の積み上げペナルティ
        hole_top_penalty = 0
        # 地面の高さ list
        highest_grounds = [-1] * self.width
        # 最も高い穴の list
        highest_holes = [-1] * self.width
        # 列ごとに切り出し
        for i in range(self.width):
            # 列取得
            col = reshape_board[:,i]
            #print(col)
            ground_level = 0
            # 上の行から 0(ブロックなし) をみつけていく, ground_level が今の列の最上層
            while ground_level < self.height and col[ground_level] == 0:
                ground_level += 1
            # その行以降の穴のlist を作り
            cols_holes = []
            for y, state in enumerate(col[ground_level + 1:]):
                # 穴のある場所をlistにする, list値として穴の位置をいれる
                if state == 0:
                    #num_holes += 1
                    cols_holes.append(self.height - (ground_level + 1 + y) - 1)
            ## 旧 1 liner 方式のカウント
            #cols_holes = [x for x in col[ground_level + 1:] if x == 0]
            # list をカウントして穴の数をカウント
            num_holes += len(cols_holes)

            # 地面の高さ配列
            highest_grounds[i] = self.height - ground_level - 1

            # 最も高い穴の位置配列
            if len(cols_holes) > 0:
                highest_holes[i] = cols_holes[0]
            else:
                highest_holes[i] = -1

        ## 最も高い穴を求める
        max_highest_hole = max(highest_holes)

        ## 到達可能の最下層より1行下の穴の位置をチェック
        if min_height > 0:
            ## 最も高いところにある穴の数
            highest_hole_num = 0
            ## 列ごとに切り出し
            for i in range(self.width):
                ## 最も高い位置の穴の列の場合
                if highest_holes[i] == max_highest_hole:
                    highest_hole_num += 1
                    ## 穴の絶対位置がhole_top_limit_heightより高く
                    ## 穴の上の地面が高いなら Penalty
                    if highest_holes[i] > self.hole_top_limit_height and \
                           highest_grounds[i] >= highest_holes[i] + self.hole_top_limit:
                        hole_top_penalty += highest_grounds[i] - (highest_holes[i])
            ## 最も高い位置にある穴の数で割る
            hole_top_penalty /= highest_hole_num
            # debug
            #print(['{:02}'.format(n) for n in highest_grounds])
            #print(['{:02}'.format(n) for n in highest_holes])
            #print(hole_top_penalty, hole_top_penalty*max_highest_hole)
            #print("==")

        return num_holes, hole_top_penalty, max_highest_hole
    
    ####################################
    # 現状状態の各種パラメータ取得 (MLP
    ####################################
    def get_state_properties(self, reshape_board):
        #削除された行の報酬
        lines_cleared, reshape_board = self.check_cleared_rows(reshape_board)
        # 穴の数
        holes, _ , _ = self.get_holes(reshape_board, -1)
        # でこぼこの数
        bumpiness, height, max_height, min_height, _ = self.get_bumpiness_and_height(reshape_board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    ####################################
    # 現状状態の各種パラメータ取得　高さ付き 今は使っていない
    ####################################
    def get_state_properties_v2(self, reshape_board):
        # 削除された行の報酬
        lines_cleared, reshape_board = self.check_cleared_rows(reshape_board)
        # 穴の数
        holes, _ , _ = self.get_holes(reshape_board, -1)
        # でこぼこの数
        bumpiness, height, max_row, min_height,_ = self.get_bumpiness_and_height(reshape_board)
        # 最大高さ
        #max_row = self.get_max_height(reshape_board)
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height, max_row])

    ####################################
    # 最大の高さを取得
    # get_bumpiness_and_height にとりこまれたので廃止
    ####################################
    def get_max_height(self, reshape_board):
        # X 軸のセルを足し算する
        sum_ = np.sum(reshape_board,axis=1)
        #print(sum_)
        row = 0
        # X 軸の合計が0になる Y 軸を探す
        while row < self.height and sum_[row] ==0:
            row += 1
        return self.height - row

    ####################################
    # 左端以外埋まっているか？
    ####################################
    def get_tetris_fill_reward(self, reshape_board):
        # 無効の場合
        if self.tetris_fill_height == 0:
            return 0

        # 報酬
        reward = 0
        # ボード上で 0 でないもの(テトリミノのあるところ)を抽出
        # (0,1,2,3,4,5,6,7) を ブロックあり True, なし False に変更
        mask = reshape_board != 0
        # X 軸のセルを足し算する
        sum_ = np.sum(mask, axis=1)
        #print(sum_)
        
        # line (1 - self.tetris_fill_height)段目の左端以外そろっているか
        for i in range(1, self.tetris_fill_height):
            # そろっている段ごとに報酬
            if self.get_line_right_fill(reshape_board, sum_, i):
                reward +=1
            ## 1段目は2倍
            if i==1:
                reward +=5

        return reward

    ####################################
    # line 段目が左端以外そろっているか
    ####################################
    def get_line_right_fill(self, reshape_board, sum_, line):
        # 1段目が端以外そろっている
        if sum_[self.height - line] == self.width -1 \
               and reshape_board[self.height - line][0] == 0 :
            #or reshape_board[self.height-1][self.width-1] == 0 ):
            #print("line:", line)
            return True
        else:
            return False

    ####################################
    #次の状態リストを取得(2次元用) DQN .... 画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
    #  get_next_func でよびだされる
    # curr_backboard 現画面
    # piece_id テトリミノ I L J T O S Z
    # currentshape_class = status["field_info"]["backboard"]
    ####################################
    def get_next_states_v2(self, curr_backboard, piece_id, CurrentShape_class):
        # 次の状態一覧
        states = {}

        # テトリミノ回転方向ごとの配置幅
        x_range_min = [0] * 4
        x_range_max = [self.width] * 4

        # 設置高さリスト drop_y_list[(direction,x)] = height
        drop_y_list = {}
        # 検証済リスト checked_board[(direction0, x0, drop_y)] =True
        checked_board = {}

        # テトリミノごとに回転数をふりわけ
        if piece_id == 5:  # O piece => 1
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7: # I, S, Z piece => 2
            num_rotations = 2
        else: # the others => 4
            num_rotations = 4

        ####################
        ## Drop Down 落下 の場合の一覧作成
        # テトリミノ回転方向ごとに一覧追加
        for direction0 in range(num_rotations):
            # テトリミノが配置できる左端と右端の座標を返す
            x0Min, x0Max = self.getSearchXRange(CurrentShape_class, direction0)
            (x_range_min[direction0], x_range_max[direction0] )= (x0Min,x0Max)

            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                # 画面ボードデータをコピーして指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
                board, drop_y = self.getBoard(curr_backboard, CurrentShape_class, direction0, x0, -1)
                # 後のため保存
                drop_y_list[(direction0, x0)] = drop_y
                checked_board[(direction0, x0, drop_y)] = True

                # ボードを２次元化
                reshape_backboard = self.get_reshape_backboard(board)
                # numpy to tensor (配列を1次元追加)
                reshape_backboard = torch.from_numpy(reshape_backboard[np.newaxis,:,:]).float()
                # 画面ボードx0で テトリミノ回転状態 direction0 に落下させたときの次の状態を作成 追加
                #  states
                #    Key = Tuple (テトリミノ Drop Down 落下 時画面ボードX座標, テトリミノ回転状態
                #                 テトリミノ Move Down 降下 数, テトリミノ追加移動X座標, テトリミノ追加回転)
                #                 ... -1 の場合 動作対象外
                #    Value = 画面ボード状態
                # (action 用)
                states[(x0, direction0, -1, -1, -1)] = reshape_backboard

        #print(len(states), end='=>')

        ## Move Down 降下無効の場合
        if self.move_down_flag == 0:
            return states

        ####################
        ## Move Down 降下 の場合の一覧作成
        # 追加補正移動
        third_y = -1
        forth_direction = -1
        fifth_x = -1
        sixth_y = -1

        # ボードを２次元化
        reshape_curr_backboard = self.get_reshape_backboard(curr_backboard)

        # ボード上で 0 でないもの(テトリミノのあるところ)を抽出
        # (0,1,2,3,4,5,6,7) を ブロックあり True, なし False に変更
        mask_board = reshape_curr_backboard != 0
        #pprint.pprint(mask_board, width = 61, compact = True)

        # 列方向 何かブロックががあれば、そのindexを返す
        # なければ画面ボード縦サイズを返す
        # 上記を 画面ボードの列に対して実施したの配列(長さ width)を返す
        invert_heights = np.where(mask_board.any(axis=0), np.argmax(mask_board, axis=0), self.height)
        # 上からの距離なので反転 (配列)
        heights = self.height - invert_heights
        ## 最大高さ
        #max_height = heights[np.argmax(heights)]
        invert_max_height = invert_heights[np.argmin(invert_heights)]

        # Debug
        if self.debug_flag_shift_rotation_success == 1 :
            print("")
        if self.debug_flag_shift_rotation == 1 or self.debug_flag_shift_rotation_success == 1 :
            print("==================================================")
            print (heights)
            print (invert_heights)
            print("first_direction:", num_rotations, " | ", CurrentShape_class.shape)

        ######## 1 回目の 回転
        for first_direction in range(num_rotations):
            if self.debug_flag_shift_rotation == 1:
                print(" 1d", first_direction,"/ second_x:",x_range_min[first_direction], " to ", x_range_max[first_direction])
            ######## 2 回目の x 軸移動
            for second_x in range(x_range_min[first_direction], x_range_max[first_direction]):
                # 高さが最大の高さ-1より大きい場合見込みがないので次へ
                if drop_y_list[(first_direction, second_x)] < invert_max_height + 1:
                    continue
                # 高さが 画面最大-2より大きい場合も見込みがないので次へ
                if invert_heights[second_x] < 2:
                    continue
                # y 座標の下限と ブロック最大の高さ-1 で検索
                if self.debug_flag_shift_rotation == 1:
                    print("   2x", second_x, "/ third_y: ",invert_max_height, " to ", drop_y_list[(first_direction, second_x)]+1)

                ######## 3 回目の y 軸降下
                for third_y in range(invert_max_height , drop_y_list[(first_direction, second_x)]+1):
                    # y 座標の下限と ブロック最大の高さ-1 で検索
                    if self.debug_flag_shift_rotation == 1:
                        print("    3y", third_y, "/ forth_direction: ")

                    # 右回転固定なので順序を変える
                    direction_order = [0] * num_rotations
                    # 最初は first_direction
                    new_direction_order = first_direction
                    # 
                    for order_num in range(num_rotations):
                        direction_order[order_num] = new_direction_order
                        new_direction_order += 1
                        if not (new_direction_order < num_rotations):
                            new_direction_order = 0

                    #print(first_direction,"::", direction_order)

                    ######## 4 回目の 回転 (Turn 2)
                    # first_direction から右回転していく
                    for forth_direction in direction_order:
                        # y 座標の下限と ブロック最大の高さ-1 で検索
                        if self.debug_flag_shift_rotation == 1:
                            print("     4d", forth_direction, "/ fifth_x: ",0, " to ", x_range_max[forth_direction] - second_x, end='')
                            print("//")
                            print("       R:", end='')
                        # 0 から探索
                        start_point_x = 0
                        # 最初と同じ回転状態ならずらしたところから探索
                        if first_direction == forth_direction:
                            start_point_x = 1

                        # 右回転判定フラグ
                        right_rotate = True
                        
                        ######## 5 回目の x 軸移動 (Turn 2)
                        # shift_x 分右にずらして確認
                        for shift_x in range(start_point_x, x_range_max[forth_direction] - second_x):
                            fifth_x = second_x + shift_x
                            # ずらした先の方が穴が深い場合は探索中止
                            if not((forth_direction, fifth_x) in drop_y_list):
                                if self.debug_flag_shift_rotation == 1:
                                    print(shift_x, ": False(OutRange) ", end='/ ')
                                break;
                            if third_y <= drop_y_list[(forth_direction, second_x + shift_x)]:
                                if self.debug_flag_shift_rotation == 1:
                                    print(shift_x, ": False(drop) ", end='/ ')
                                break;
                            # direction (回転状態)のテトリミノ2次元座標配列を取得し、それをx,yに配置した場合の座標配列を返す
                            coordArray = self.getShapeCoordArray(CurrentShape_class, forth_direction, fifth_x, third_y)
                            # x座標方向にテトリミノが動かせるか確認する
                            judge = self.try_move_(curr_backboard, coordArray)
                            if self.debug_flag_shift_rotation == 1:
                                print(shift_x, ":", judge, end='/')
                            # 右移動可能
                            if judge:
                                ####
                                ## 登録済か確認し、STATES へ入れる
                                states, checked_board = \
                                    self.second_drop_down(curr_backboard, CurrentShape_class, \
                                                          first_direction, second_x, third_y, forth_direction, fifth_x, \
                                                          states, checked_board)
                            # 右移動不可なら終了
                            else:
                                ## 最初の位置で右回転がうまく行かない場合は回転作業も抜ける
                                if shift_x == 0 and judge == False:
                                    right_rotate = False
                                break;

                        ## 最初の位置で右回転がうまく行かない場合は抜ける
                        if right_rotate == False:
                            if self.debug_flag_shift_rotation_success == 1:
                                print(" |||", CurrentShape_class.shape, "-", forth_direction,\
                                      "(", second_x, ",", third_y, ")|||<=RotateStop|||")
                            break;

                        ## 右ずらし終了
                        if self.debug_flag_shift_rotation == 1:
                            print("//")
                            print("       L:", end='')

                        ########
                        # shift_x を左にずらして確認
                        for shift_x in range(-1, -second_x - 1, -1):
                            fifth_x = second_x + shift_x
                            # ずらした先の方が穴が深い場合は探索中止
                            if not((forth_direction, fifth_x) in drop_y_list):
                                break;
                            if third_y <= drop_y_list[(forth_direction, fifth_x)]:
                                break;
                            # direction (回転状態)のテトリミノ2次元座標配列を取得し、それをx,yに配置した場合の座標配列を返す
                            coordArray = self.getShapeCoordArray(CurrentShape_class, forth_direction, fifth_x, third_y)
                            # x座標方向にテトリミノが動かせるか確認する
                            judge = self.try_move_(curr_backboard, coordArray)

                            if self.debug_flag_shift_rotation == 1:
                                print(shift_x, ":", judge, end='/ ')

                            # 左移動可能
                            if judge:
                                ####
                                ## 登録済か確認し、STATES へ入れる
                                states, checked_board = \
                                    self.second_drop_down(curr_backboard, CurrentShape_class, \
                                                          first_direction, second_x, third_y, forth_direction, fifth_x, \
                                                          states, checked_board)

                            # 左移動不可なら終了
                            else:
                                break;
                        ## 左ずらし終了
                        if self.debug_flag_shift_rotation == 1:
                            print("//")
                        #end shift_x
                    #end forth
                #end third
            #end second
        #end first

        # Debug
        if self.debug_flag_shift_rotation_success == 1 :
            print("")
        #print (len(states))
        ## states (action) を返す
        return states


    ####################################
    #次の状態を取得(1次元用) MLP  .... 画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
    #  get_next_func でよびだされる
    ####################################
    def get_next_states(self, curr_backboard, piece_id, CurrentShape_class):
        # 次の状態一覧
        states = {}

        if piece_id == 5:  # O piece
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7:
            num_rotations = 2
        else:
            num_rotations = 4

        ####################
        ## Drop Down 落下 の場合の一覧作成
        # テトリミノ回転方向ごとに一覧追加
        for direction0 in range(num_rotations):
            # テトリミノが配置できる左端と右端の座標を返す
            x0Min, x0Max = self.getSearchXRange(CurrentShape_class, direction0)
            # テトリミノ左端から右端まで配置
            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                # 画面ボードデータをコピーして指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
                board, drop_y = self.getBoard(curr_backboard, CurrentShape_class, direction0, x0, -1)
                #ボードを２次元化
                reshape_board = self.get_reshape_backboard(board)
                # 画面ボードx0で テトリミノ回転状態 direction0 に落下させたときの次の状態を作成
                #  states
                #    Key = Tuple (テトリミノ Drop Down 落下 時画面ボードX座標, テトリミノ回転状態
                #                 テトリミノ Move Down 降下 数, テトリミノ追加移動X座標, テトリミノ追加回転)
                #    Value = 画面ボード状態
                states[(x0, direction0, 0, 0, 0)] = self.get_state_properties(reshape_board)

        return states

    ####################################
    # テトリミノ配置して states に登録する (ずらし時用)
    ####################################
    def second_drop_down(self, curr_backboard, CurrentShape_class, \
                         first_direction, second_x, third_y, forth_direction, fifth_x, states, checked_board):
        ##debug
        #self.debug_flag_drop_down = 1

        # 画面ボードデータをコピーして指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
        new_board, drop_y = self.getBoard(curr_backboard, CurrentShape_class, forth_direction, fifth_x, third_y)
        # 落下指定
        sixth_y = third_y + drop_y

        #debug
        if self.debug_flag_shift_rotation_success == 1:
            print(" ***", CurrentShape_class.shape, "-", forth_direction,\
                  "(", fifth_x, ",", third_y,"=>", sixth_y, ")***", end='')

        # 登録済でないか確認
        if not ((forth_direction, fifth_x, sixth_y) in checked_board):
            #debug
            if self.debug_flag_shift_rotation_success == 1:
                print("<=NEW***", end='')
            # 降下後動作として登録 (重複防止)
            checked_board[(forth_direction, fifth_x, sixth_y)] = True
            # ボードを２次元化
            reshape_backboard = self.get_reshape_backboard(new_board)
            # numpy to tensor (配列を1次元追加)
            reshape_backboard = torch.from_numpy(reshape_backboard[np.newaxis,:,:]).float()
            ####################
            # 画面ボードx0で テトリミノ移動した時次のの状態を作成 追加
            #  states
            #    Key = Tuple (テトリミノ Drop Down 落下 時画面ボードX座標, テトリミノ回転状態
            #                 テトリミノ Move Down 降下 数, テトリミノ追加移動X座標, テトリミノ追加回転)
            #                 ... -1 の場合 動作対象外
            #    Value = 画面ボード状態
            # (action 用)
            states[(second_x, first_direction, third_y, forth_direction, fifth_x)] = reshape_backboard


        #debug
        if self.debug_flag_shift_rotation_success == 1:
            print("")

        return states, checked_board

    ####################################
    # 配置できるか確認する
    # board: 1次元座標
    # coordArray: テトリミノ2次元座標
    ####################################
    def try_move_(self, board, coordArray):
        # テトリミノ座標配列(各マス)ごとに
        judge = True

        debug_board = [0] * self.width * self.height
        debug_log = "";

        for coord_x, coord_y in coordArray:
            debug_log = debug_log+ "==("+ str(coord_x)+ ","+ str(coord_y)+ ") "

            # テトリミノ座標coord_y が 画面下限より上　かつ　(テトリミノ座標coord_yが画面上限より上 
            # テトリミノ座標coord_x, テトリミノ座標coord_yのブロックがない)
            if 0 <= coord_x and \
                   coord_x < self.width and \
                   coord_y < self.height and \
                   (coord_y * self.width + coord_x < 0 or \
                   board[coord_y * self.width + coord_x] == 0):

                # はまる
                debug_board [coord_y * self.width + coord_x] = 1

            # はまらないので False
            else:
                judge = False
                # はまらない
                #self.debug_flag_try_move = 1
                if 0 <= coord_x and coord_x < self.width \
                   and 0 <= coord_y and coord_y < self.height:
                    debug_board [coord_y * self.width + coord_x ] = 8

        # Debug 用
        if self.debug_flag_try_move == 1:
            print( debug_log)
            pprint.pprint(board, width = 31, compact = True)
            pprint.pprint(debug_board, width = 31, compact = True)
            self.debug_flag_try_move =0
        return judge


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
    def step_v2(self, curr_backboard, action, curr_shape_class):
        x0, direction0, third_y, forth_direction, fifth_x = action
        ## 画面ボードデータをコピーして指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
        board, drop_y = self.getBoard(curr_backboard, curr_shape_class, direction0, x0, -1)
        ##ボードを２次元化
        reshape_board = self.get_reshape_backboard(board)
        #### 報酬計算元の値取得
        ## でこぼこ度, 高さ合計, 高さ最大, 高さ最小を求める
        bampiness, total_height, max_height, min_height, left_side_height = self.get_bumpiness_and_height(reshape_board)
        #max_height = self.get_max_height(reshape_board)
        ## 穴の数, 穴の上積み上げ Penalty, 最も高い穴の位置を求める
        hole_num, hole_top_penalty, max_highest_hole = self.get_holes(reshape_board, min_height)
        ## 左端あけた形状の報酬計算
        tetris_reward = self.get_tetris_fill_reward(reshape_board)
        ## 消せるセルの確認
        lines_cleared, reshape_board = self.check_cleared_rows(reshape_board)
        ## 報酬の計算
        reward = self.reward_list[lines_cleared] * (1 + (self.height - max(0,max_height))/self.height_line_reward)
        ## 継続報酬
        #reward += 0.01
        #### 形状の罰報酬
        ## でこぼこ度罰
        reward -= self.reward_weight[0] * bampiness 
        ## 最大高さ罰
        if max_height > self.max_height_relax:
            reward -= self.reward_weight[1] * max(0,max_height)
        ## 穴の数罰
        reward -= self.reward_weight[2] * hole_num
        ## 穴の上のブロック数罰
        reward -= self.hole_top_limit_reward * hole_top_penalty * max_highest_hole
        ## 左端以外埋めている状態報酬
        reward += tetris_reward * self.tetris_fill_reward
        ## 左端が高すぎる場合の罰
        if left_side_height > self.bumpiness_left_side_relax:
            reward -= (left_side_height - self.bumpiness_left_side_relax) * self.left_side_height_penalty

        self.epoch_reward += reward 

        # スコア計算
        self.score += self.score_list[lines_cleared]
        # 消去ラインカウント
        self.cleared_lines += lines_cleared
        self.cleared_col[lines_cleared] += 1
        # テトリミノ数カウント増やす
        self.tetrominoes += 1
        return reward

    ####################################
    #報酬を計算(1次元用) 
    #reward_func から呼び出される
    ####################################
    def step(self, curr_backboard, action, curr_shape_class):
        x0, direction0, third_y, forth_direction, fifth_x = action
        # 画面ボードデータをコピーして指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
        board, drop_y = self.getBoard(curr_backboard, curr_shape_class, direction0, x0, -1)
        #ボードを２次元化
        reshape_board = self.get_reshape_backboard(board)
        # 報酬計算元の値取得
        bampiness, height, max_height, min_height, _ = self.get_bumpiness_and_height(reshape_board)
        #max_height = self.get_max_height(reshape_board)
        hole_num, _ , _ = self.get_holes(reshape_board, min_height)
        lines_cleared, reshape_board = self.check_cleared_rows(reshape_board)
        #### 報酬の計算
        reward = self.reward_list[lines_cleared] 
        # 継続報酬
        #reward += 0.01
        # 罰
        reward -= self.reward_weight[0] *bampiness 
        if max_height > self.max_height_relax:
            reward -= self.reward_weight[1] * max(0,max_height)
        reward -= self.reward_weight[2] * hole_num
        self.epoch_reward += reward

        # スコア計算
        self.score += self.score_list[lines_cleared]

        # 消した数追加
        self.cleared_lines += lines_cleared
        self.cleared_col[lines_cleared] += 1
        self.tetrominoes += 1
        return reward

    ####################################
    ####################################
    ####################################
    ####################################
    # 次の動作取得: ゲームコントローラから毎回呼ばれる
    ####################################
    ####################################
    ####################################
    ####################################
    def GetNextMove(self, nextMove, GameStatus, yaml_file=None,weight=None):

        t1 = datetime.now()
        # RESET 関数設定 callback function 代入 (Game Over 時)
        nextMove["option"]["reset_callback_function_addr"] = self.update
        # mode の取得 (train である) 
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

        #reshape_backboard = self.get_reshape_backboard(curr_backboard)
        #print(reshape_backboard)
        #self.state = reshape_backboard

        ###############################################
        ## Move Down で 前回の持ち越し動作がある場合　その動作をして終了
        if self.skip_drop != [-1, -1, -1]:
            # third_y, forth_direction, fifth_x
            nextMove["strategy"]["direction"] = self.skip_drop[1]
            # 横方向
            nextMove["strategy"]["x"] = self.skip_drop[2]
            # Move Down 降下
            nextMove["strategy"]["y_operation"] = 1
            # Move Down 降下 数
            nextMove["strategy"]["y_moveblocknum"] = 1
            # 前のターンで Drop をスキップしていたか？を解除 (-1: していない, それ以外: していた)
            self.skip_drop = [-1, -1, -1]
            ## 終了時刻
            if self.time_disp:
                print(datetime.now()-t1)
            ## 終了   
            return nextMove
        
        ###################
        #画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
        # next_steps
        #    Key = Tuple (テトリミノ画面ボードX座標, テトリミノ回転状態)
        #                 テトリミノ Move Down 降下 数, テトリミノ追加移動X座標, テトリミノ追加回転)
        #    Value = 画面ボード状態
        next_steps = self.get_next_func(curr_backboard, curr_piece_id, curr_shape_class)
        #print (len(next_steps), end='=>')

        ###############################################
        ###############################################
        # 学習の場合
        ###############################################
        ###############################################
        if self.mode=="train" or self.mode=="train_sample" or self.mode=="train_sample2":
            #### init parameter
            # epsilon = 学習結果から乱数で変更する割合対象
            # num_decay_epochs より前までは比例で初期 epsilon から減らしていく
            # num_decay_ecpchs 以降は final_epsilon固定
            epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.epoch, 0) * (
                    self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
            u = random()
            # epsilon より乱数 u が小さい場合フラグをたてる
            random_action = u <= epsilon

            # 次のテトリミノ予測
            if self.predict_next_num_train > 0:
                ##########################
                # モデルの学習実施
                ##########################
                self.model.train()
                # index_list [1番目index, 2番目index, 3番目index ...] => q
                index_list = []
                # index_list_to_q (1番目index, 2番目index, 3番目index ...) => q
                index_list_to_q = {}
                ######################
                # 次の予測を上位predict_next_steps_trainつ実施, 1番目からpredict_next_num_train番目まで予測
                index_list, index_list_to_q, next_actions, next_states \
                            = self.get_predictions(self.model, True, GameStatus, next_steps, self.predict_next_steps_train, 1, self.predict_next_num_train, index_list, index_list_to_q, -60000)
                #print(index_list_to_q)
                #print("max")

                # 全予測の最大 q
                max_index_list = max(index_list_to_q, key=index_list_to_q.get)
                #print(max(index_list_to_q, key=index_list_to_q.get))
                #print(max_index_list[0].item())
                #print (len(next_steps))
                #print("============================")
                # 乱数が epsilon より小さい場合は
                if random_action:
                    # index を乱数とする
                    index = randint(0, len(next_steps) - 1)
                else:
                    # 1手目の index 入手
                    index = max_index_list[0].item()
            else:
                # 次の状態一覧の action と states で配列化
                #    next_actions  = Tuple (テトリミノ画面ボードX座標, テトリミノ回転状態)　一覧
                #    next_states = 画面ボード状態 一覧
                next_actions, next_states = zip(*next_steps.items())
                # next_states (画面ボード状態 一覧) のテンソルを連結 (画面ボード状態のlist の最初の要素に状態が追加された)
                next_states = torch.stack(next_states)

                ## GPU 使用できるときは使う
#                if torch.cuda.is_available():
#                    next_states = next_states.cuda()
            
                ##########################
                # モデルの学習実施
                ##########################
                self.model.train()
                # テンソルの勾配の計算を不可とする(Tensor.backward() を呼び出さないことが確実な場合)
                with torch.no_grad():
                    # 順伝搬し Q 値を取得 (model の __call__ ≒ forward)
                    predictions = self.model(next_states)[:, 0]
                    # predict = self.model(next_states)[:,:]
                    # predictions = predict[:,0]
                    # print("input: ", next_states)
                    # print("predict: ", predict[:,0])

                # 乱数が epsilon より小さい場合は
                if random_action:
                    # index を乱数とする
                    index = randint(0, len(next_steps) - 1)
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
            reward = self.reward_func(curr_backboard, action, curr_shape_class)
            
            done = False #game over flag
            
            #####################################
            # Double DQN 有効時
            #======predict max_a Q(s_(t+1),a)======
            #if use double dqn, predicted by main model
            if self.double_dqn:
                # 画面ボードデータをコピーして 指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
                next_backboard, drop_y  = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0], action[2])
                #画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
                next2_steps = self.get_next_func(next_backboard, next_piece_id, next_shape_class)
                # 次の状態一覧の action と states で配列化
                next2_actions, next2_states = zip(*next2_steps.items())
                # next_states のテンソルを連結
                next2_states = torch.stack(next2_states)
                ## GPU 使用できるときは使う
#                if torch.cuda.is_available():
#                    next2_states = next2_states.cuda()
                ##########################
                # モデルの学習実施
                ##########################
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
            #if use target net, predicted by target model
            elif self.target_net:
                # 画面ボードデータをコピーして 指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
                next_backboard, drop_y  = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0], action[2])
                #画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
                next2_steps =self.get_next_func(next_backboard, next_piece_id, next_shape_class)
                # 次の状態一覧の action と states で配列化
                next2_actions, next2_states = zip(*next2_steps.items())
                # next_states のテンソルを連結
                next2_states = torch.stack(next2_states)
                ## GPU 使用できるときは使う
#                if torch.cuda.is_available():
#                    next2_states = next2_states.cuda()
                ##########################
                # モデルの学習実施
                ##########################
                self.target_model.train()
                # テンソルの勾配の計算を不可とする
                with torch.no_grad():
                    #### 『ターゲットモデル』で Q値算出
                    next_predictions = self.target_model(next2_states)[:, 0]
                # 次の index を推論の最大値とする
                next_index = torch.argmax(next_predictions).item()
                # 次の状態を index で指定し取得
                next2_state = next2_states[next_index, :]
                
            #if not use target net,predicted by main model
            else:
                # 画面ボードデータをコピーして 指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
                next_backboard, drop_y  = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0], action[2])
                #画面ボードで テトリミノ回転状態 に落下させたときの次の状態一覧を作成
                next2_steps =self.get_next_func(next_backboard, next_piece_id, next_shape_class)
                # 次の状態一覧の action と states で配列化
                next2_actions, next2_states = zip(*next2_steps.items())
                # 次の状態を index で指定し取得
                next2_states = torch.stack(next2_states)

                ## GPU 使用できるときは使う
#                if torch.cuda.is_available():
#                    next2_states = next2_states.cuda()
                ##########################
                # モデルの学習実施
                ##########################
                self.model.train()
                # テンソルの勾配の計算を不可とする
                with torch.no_grad():
                    # 順伝搬し Q 値を取得 (model の __call__ ≒ forward)
                    next_predictions = self.model(next2_states)[:, 0]

                # epsilon = 学習結果から乱数で変更する割合対象
                # num_decay_epochs より前までは比例で初期 epsilon から減らしていく
                # num_decay_ecpchs 以降は final_epsilon固定
                epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.epoch, 0) * (
                self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
                u = random()
                # epsilon より乱数 u が小さい場合フラグをたてる
                random_action = u <= epsilon
                
                # 乱数が epsilon より小さい場合は
                if random_action:
                    # index を乱数指定
                    next_index = randint(0, len(next2_steps) - 1)
                else:
                   # 次の index を推論の最大値とする
                    next_index = torch.argmax(next_predictions).item()
                # 次の状態を index により指定
                next2_state = next2_states[next_index, :]

            #=======================================
            # Episode Memory に
            # next_state  次の候補第1位手
            # reward 報酬
            # next2_state 比較対象のモデルによる候補手 (Target net など)
            # done Game Over flag
            #self.replay_memory.append([next_state, reward, next2_state,done])
            self.episode_memory.append([next_state, reward, next2_state, done])
            # 優先順位つき経験学習有効ならば
            if self.prioritized_replay:
                # キューにリプレイ用の情報を格納していく
                self.PER.store()
            
            #self.replay_memory.append([self.state, reward, next_state,done])

            ###############################################
            ## 学習時 次の動作指定
            ###############################################
            ## 前のターンで Drop をスキップしていたか？ (-1: していない, それ以外: していた)
            # third_y, forth_direction, fifth_x
            #self.skip_drop = [-1, -1, -1]
        
            # テトリミノ回転
            nextMove["strategy"]["direction"] = action[1]
            # 横方向
            nextMove["strategy"]["x"] = action[0]
            ###########
            # Drop Down 落下の場合
            if action[2] == -1 and action[3] == -1 and action[4] == -1:
                # Drop Down 落下
                nextMove["strategy"]["y_operation"] = 1
                # Move Down 降下数
                nextMove["strategy"]["y_moveblocknum"] = 1
                # 前のターンで Drop をスキップしていたか？ (-1: していない, それ以外: していた)
                self.skip_drop = [-1, -1, -1]
            ###########
            # Move Down 降下の場合
            else:
                # Move Down 降下
                nextMove["strategy"]["y_operation"] = 0
                # Move Down 降下 数
                nextMove["strategy"]["y_moveblocknum"] = action[2]
                # 前のターンで Drop をスキップしていたか？ (-1: していない, それ以外: していた)
                # third_y, forth_direction, fifth_x
                self.skip_drop = [action[2], action[3], action[4]]
                # debug
                if self.debug_flag_move_down == 1:
                    print("Move Down: ", "(", action[0], ",", action[2], ")")

            ##########
            # 学習終了処理
            ##########
            # 1ゲーム(EPOCH)の上限テトリミノ数を超えたらリセットフラグを立てる
            if self.tetrominoes > self.max_tetrominoes:
                nextMove["option"]["force_reset_field"] = True
            # STATE = next_state 代入
            self.state = next_state

        ###############################################
        ###############################################
        # 推論 の場合
        ###############################################
        ###############################################
        elif self.mode == "predict" or self.mode == "predict_sample":
            ##############
            # model 切り替え
            if self.weight2_available:
                #ボードを２次元化
                reshape_board = self.get_reshape_backboard(curr_backboard)
                ## 最も高い穴の位置を求める
                _ , _ , max_highest_hole = self.get_holes(reshape_board, -1)
                ## model2 切り替え条件
                if max_highest_hole < self.predict_weight2_enable_index:
                    self.weight2_enable = True
                ## model1 切り替え条件
                if max_highest_hole > self.predict_weight2_disable_index:
                    self.weight2_enable = False

                #debug
                print (GameStatus["judge_info"]["block_index"], self.weight2_enable, max_highest_hole)


            ##############
            # model 指定
            predict_model = self.model
            if self.weight2_enable:
                predict_model = self.model2

            #推論モードに切り替え
            predict_model.eval()

            # 次のテトリミノ予測
            if self.predict_next_num > 0:
                # index_list [1番目index, 2番目index, 3番目index ...] => q
                index_list = []
                # index_list_to_q (1番目index, 2番目index, 3番目index ...) => q
                index_list_to_q = {}
                ######################
                # 次の予測を上位predict_next_stepsつ実施, 1番目からpredict_next_num番目まで予測
                index_list, index_list_to_q, next_actions, next_states \
                            = self.get_predictions(predict_model, False, GameStatus, next_steps, self.predict_next_steps, 
                                1, self.predict_next_num, index_list, index_list_to_q, -60000)
                #print(index_list_to_q)
                #print("max")

                # 全予測の最大 q
                max_index_list = max(index_list_to_q, key=index_list_to_q.get)
                #print(max(index_list_to_q, key=index_list_to_q.get))
                #print(max_index_list[0].item())
                #print("============================")
                # 1手目の index 入手
                index = max_index_list[0].item()

            else:
                ### 画面ボードの次の状態一覧を action と states にわけ、states を連結
                next_actions, next_states = zip(*next_steps.items())
                next_states = torch.stack(next_states)
                ## 順伝搬し Q 値を取得 (model の __call__ ≒ forward)
                predictions = predict_model(next_states)[:, 0]
                ## 最大値の index 取得
                index = torch.argmax(predictions).item()

            # 次の action を index を元に決定
            # 0: 2番目 X軸移動
            # 1: 1番目 テトリミノ回転
            # 2: 3番目 Y軸降下 (-1: で Drop)
            # 3: 4番目 テトリミノ回転 (Next Turn)
            # 4: 5番目 X軸移動 (Next Turn)
            action = next_actions[index]

            ###############################################
            ## 推論時 次の動作指定
            ###############################################
            ## 前のターンで Drop をスキップしていたか？ (-1: していない, それ以外: していた)
            # third_y, forth_direction, fifth_x
            #self.skip_drop = [-1, -1, -1]
            # テトリミノ回転
            nextMove["strategy"]["direction"] = action[1]
            # 横方向
            nextMove["strategy"]["x"] = action[0]
            ###########
            # Drop Down 落下の場合
            if action[2] == -1 and action[3] == -1 and action[4] == -1:
                # Drop Down 落下
                nextMove["strategy"]["y_operation"] = 1
                # Move Down 降下数
                nextMove["strategy"]["y_moveblocknum"] = 1
                # 前のターンで Drop をスキップしていたか？ (-1: していない, それ以外: していた)
                self.skip_drop = [-1, -1, -1]
            ###########
            # Move Down 降下の場合
            else:
                # Move Down 降下
                nextMove["strategy"]["y_operation"] = 0
                # Move Down 降下 数
                nextMove["strategy"]["y_moveblocknum"] = action[2]
                # 前のターンで Drop をスキップしていたか？ (-1: していない, それ以外: していた)
                # third_y, forth_direction, fifth_x
                self.skip_drop = [action[2], action[3], action[4]]
                # debug
                if self.debug_flag_move_down == 1:
                    print("Move Down: ", "(", action[0], ",", action[2], ")")
        ## 終了時刻
        if self.time_disp:
            print(datetime.now()-t1)
        ## 終了
        return nextMove

    ####################################
    # テトリミノの予告に対して次の状態リストをTop num_steps個取得
    # self: 
    # predict_model: モデル指定
    # is_train: 学習モード状態の場合 (no_gradにするため)
    # GameStatus: GameStatus
    # prev_steps: 前の手番の候補手リスト
    # num_steps: 1つの手番で候補手をいくつ探すか
    # next_order: いくつ先の手番か
    # left: 何番目の手番まで探索するか
    # index_list: 手番ごとのindexリスト
    # index_list_to_q: 手番ごとのindexリストから Q 値への変換
    ####################################
    def get_predictions(self, predict_model, is_train, GameStatus, prev_steps, num_steps, next_order, left, index_list, index_list_to_q, highest_q):
        ## 次の予測一覧
        next_predictions = []
        ## index_list 複製
        new_index_list = []

        ## 予測の画面ボード
        #next_predict_backboard = []

        # 画面ボードの次の状態一覧を action と states にわけ、states を連結
        next_actions, next_states = zip(*prev_steps.items())
        next_states = torch.stack(next_states)
        # 学習モードの場合
        if is_train:
            ## GPU 使用できるときは使う
#            if torch.cuda.is_available():
#                next_states = next_states.cuda()
            # テンソルの勾配の計算を不可とする
            with torch.no_grad():
                # 順伝搬し Q 値を取得 (model の __call__ ≒ forward)
                predictions = predict_model(next_states)[:, 0]
        # 推論モードの場合
        else:
            # 順伝搬し Q 値を取得 (model の __call__ ≒ forward)
            predictions = predict_model(next_states)[:, 0]

        ## num_steps 番目まで Top の index 取得
        top_indices = torch.topk(predictions, num_steps).indices

        # 再帰探索
        if next_order < left:
            ## 予測の画面ボード取得
            #predict_order = 0
            for index in top_indices:
                # index_list に追加
                new_index_list = index_list.copy()
                new_index_list.append(index)
                # Q 値比較
                now_q = predictions[index].item()
                if now_q > highest_q:
                    # 最高値とする
                    highest_q = now_q

                # 次の画面ボード (torch) をひっぱってくる
                next_state = next_states[index, :]
                #print(next_order, ":", next_state)
                # Numpy に変換し int にして、1次元化
                #next_predict_backboard.append(np.ravel(next_state.numpy().astype(int)))
                #print(predict_order,":", next_predict_backboard[predict_order])
        
                # 次の予想手リスト
                # next_state Numpy に変換し int にして、1次元化
                next_steps = self.get_next_func( np.ravel(next_state.numpy().astype(int)),
                                     GameStatus["block_info"]["nextShapeList"]["element"+str(next_order)]["index"],
                                     GameStatus["block_info"]["nextShapeList"]["element"+str(next_order)]["class"]) 
                #GameStatus["block_info"]["nextShapeList"]["element"+str(1)]["direction_range"]
    
                ## 次の予測を上位 num_steps 実施, next_order 番目から left 番目まで予測
                new_index_list, index_list_to_q, new_next_actions, new_next_states \
                                = self.get_predictions(predict_model, is_train, GameStatus, 
                                    next_steps, num_steps, next_order+1, left, new_index_list, index_list_to_q, highest_q)
                # 次のカウント
                #predict_order += 1
        # 再帰終了
        else:
            # Top のみ index_list に追加
            new_index_list = index_list.copy()
            new_index_list.append(top_indices[0])
            # Q 値比較
            now_q = predictions[top_indices[0]].item()
            if now_q > highest_q:
                # 最高値とする
                highest_q = now_q
            # index_list から q 値への辞書をつくる
            #print (new_index_list, highest_q, now_q)
            index_list_to_q[tuple(new_index_list)] = highest_q


        ## 次の予測一覧とQ値, および最初の action, state を返す
        return new_index_list, index_list_to_q, next_actions, next_states

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
        # direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置した場合の2次元座標配列を返す
        coordArray = Shape_class.getCoords(direction, x, y) # get array from shape direction, x, y.
        return coordArray

    ####################################
    # 画面ボードデータをコピーして指定座標にテトリミノを配置し落下させた画面ボードとy座標を返す
    # board_backboard: 現状画面ボード
    # Shape_class: テトリミノ現/予告リスト
    # direction: テトリミノ回転方向
    # center_x: テトリミノx座標
    # center_y: テトリミノy座標
    ####################################
    def getBoard(self, board_backboard, Shape_class, direction, center_x, center_y):
        # 
        # get new board.
        #
        # copy backboard data to make new board.
        # if not, original backboard data will be updated later.
        board = copy.deepcopy(board_backboard)
        # 指定座標から落下させたところにテトリミノを固定しその画面ボードを返す
        _board, drop_y = self.dropDown(board, Shape_class, direction, center_x, center_y)
        return _board, drop_y

    ####################################
    # 指定座標から落下させたところにテトリミノを固定しその画面ボードを返す
    # board: 現状画面ボード
    # Shape_class: テトリミノ現/予告リスト
    # direction: テトリミノ回転方向
    # center_x: テトリミノx座標
    # center_y: テトリミノy座標 (-1: Drop 指定)
    ####################################
    def dropDown(self, board, Shape_class, direction, center_x, center_y):
        # 
        # internal function of getBoard.
        # -- drop down the shape on the board.
        #
        ###############
        ## Drop Down 落下の場合
        if center_y == -1:
            center_y = 0

        # 画面ボード下限座標として dy 設定
        dy = self.board_data_height - 1
        # direction (回転状態)のテトリミノ2次元座標配列を取得し、それをx,yに配置した場合の座標配列を返す
        coordArray = self.getShapeCoordArray(Shape_class, direction, center_x, center_y)

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
        # dy: テトリミノ落下下限座標を指定
        _board = self.dropDownWithDy(board, Shape_class, direction, center_x, dy)

        # debug
        if self.debug_flag_drop_down == 1:
            print("<%%", direction, center_x, center_y, dy, "%%>", end='')
            self.debug_flag_drop_down = 0
        return _board, dy

    ####################################
    # 指定位置にテトリミノを固定する
    # board: 現状画面ボード
    # Shape_class: テトリミノ現/予告リスト
    # direction: テトリミノ回転方向
    # center_x: テトリミノx座標
    # center_y: テトリミノy座標を指定
    ####################################
    def dropDownWithDy(self, board, Shape_class, direction, center_x, center_y):
        #
        # internal function of dropDown.
        #
        # board コピー
        _board = board
        # direction (回転状態)のテトリミノ2次元座標配列を取得し、それをx,yに配置した場合の座標配列を返す
        coordArray = self.getShapeCoordArray(Shape_class, direction, center_x, 0)
        # テトリミノ座標配列を順に進める
        for _x, _y in coordArray:
            #center_x, center_y の 画面ボードにブロックを配置して、その画面ボードデータを返す
            _board[(_y + center_y) * self.board_data_width + _x] = Shape_class.shape
        return _board
BLOCK_CONTROLLER_TRAIN = Block_Controller()
