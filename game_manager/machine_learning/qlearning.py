
from collections import deque
import numpy as np
import copy
import torch

################################################
################################################
# 優先順位つき経験学習クラス
################################################
################################################
class PRIORITIZED_EXPERIENCE_REPLAY():
    ####################################
    # 初期化
    ####################################
    def __init__(self,N ,alpha=0.7,beta=0.5,gamma=0.99,mode="rank"):
        # replay_memory_size の queue を作成。deque で maxlen 指定して自動的にあふれると削除
        self.replay_priority_queue = deque(maxlen=N)
        # index は 1 から順に番号
        self.replay_index = [i for i in range(N)]
        # weights は全部1
        self.weights = [1 for i in range(N)]
        #
        self.alpha = alpha
        #
        self.beta = beta
        # Rank モードならランク付け
        self.mode = mode
        #
        self.gamma = gamma


    ####################################
    # 優先順位キューを埋める
    # memoryが埋まるまでは1固定、その後は最大値をいれる
    ####################################
    def store(self):
        #print("store ..")
        # メモリサイズが 0 なら 1 追加
        if len(self.replay_priority_queue)==0:
            self.replay_priority_queue.append(1.0) 
        # そうでないならその時の queue の最大値を追加
        else:
            max_priority = max(self.replay_priority_queue)
            self.replay_priority_queue.append(max_priority) 

    ####################################
    # alpha 乗して正規化 (rank_based_priority で使用)
    ####################################
    def normalize(self,replay_priority):
        replay_priority = replay_priority ** self.alpha
        sum_priority = np.sum(replay_priority)
        replay_priority = replay_priority/sum_priority
        return replay_priority
    
    ####################################
    # リプレイ優先度の決定 優先度低いものを 1/n にする (mode "rank")
    ####################################
    def rank_based_priority(self,replay_priority):
        replay_priority_index = np.argsort(replay_priority)[::-1]
        # ランクごとに優先度を 1/n にする
        for i,index in enumerate(replay_priority_index):
            replay_priority[index] =  1.0 / (i+1.0)
        # α乗して正規化
        replay_priority = self.normalize(replay_priority)
        return replay_priority
        
    ####################################
    # リプレイメモリの batch を優先度に基づきサンプリング
    ####################################
    def sampling(self,replay_memory,batch_size):
        # リプレイ優先度取得
        replay_priority = np.array(copy.copy(self.replay_priority_queue))
        replay_priority = replay_priority[:len(replay_memory)]
        if self.mode=="rank":
            replay_priority = self.rank_based_priority(replay_priority)
        # リプレイ index 取得
        replay_index = self.replay_index[:len(replay_priority)]
        # 優先順位つきランダム取得
        replay_batch_index = np.random.choice(replay_index,batch_size,p=replay_priority)
        try:
            replay_batch = deque([replay_memory[j] for j in replay_batch_index])
        except IndexError:
            print(replay_memory)
            print(len(replay_memory))
        
        N = len(replay_priority)
        # リプレイ優先度から重みづけ計算
        for i in range(len(replay_priority)):
            self.weights[i] = (N*replay_priority[i])**(-self.beta)
        max_weights = max(self.weights)
        self.weights /= max_weights
        return replay_batch,replay_batch_index
    
    ####################################
    # 報酬から優先度更新
    ####################################
    def update_priority(self,replay_batch_index,reward_batch,q_batch,next_q_batch):
        memo = []
        weights = []
        for i,index in enumerate(replay_batch_index):
            # 重みづけに リプレイバッチに抽出された index 追加
            weights.append(self.weights[index])
            if index in memo:
                continue
            # index 新規なら memo に追加
            memo.append(index )
            # 勾配計算なしで誤差計算 (割引率γおりこみ)
            with torch.no_grad():
                #print(self.gamma *next_q_batch[i] - q_batch[i])
                TD_error = float(reward_batch[i] + self.gamma *next_q_batch[i] - q_batch[i])
            self.replay_priority_queue[index] = abs(TD_error)
        # Numpy にもどす
        weights  = np.array(weights,dtype=np.float64)
        #print(self.replay_priority_queue)
        return torch.from_numpy(weights)
        

################################################
################################################
# マルチステップ学習クラス
################################################
################################################
class Multi_Step_Learning:
    ####################################
    # 
    ####################################
    def __init__(self,step_num=3,gamma=0.99):
        self.step_num = step_num
        self.gamma = gamma

    ####################################
    # 
    ####################################
    def __get_mult_step(self,episode,start_index,end_index):
        coefficient = 1.0/(self.gamma**2)
        reward = 0
        episode_size = len(episode)
        for k in range(start_index,end_index):
            coefficient *=self.gamma
            if k<episode_size:
                reward += episode[k][1]*coefficient
                next_state =  episode[k][2]
        return reward,next_state

    ####################################
    # 
    ####################################
    def arrange(self,episode):
        #The batch muste be [state,reward,next_state, done].
        for i in range(len(episode)):
            episode[i][1],episode[i][2] = self.__get_mult_step(episode,i,i+self.step_num)

        return episode

    ####################################
    # 
    ####################################
    def get_y_batch(self,done_batch,reward_batch, next_prediction_batch):
        return torch.cat(tuple(
                        reward if done[0] 
                        else self.gamma*reward + (self.gamma**self.step_num) * prediction for done ,reward, prediction 
                        in zip(done_batch,reward_batch, next_prediction_batch)))[:, None]     