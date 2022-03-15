
from collections import deque
import numpy as np
import copy
import torch

class PRIORITIZED_EXPERIENCE_REPLAY():
    def __init__(self,N ,alpha=0.7,beta=0.5,gamma=0.99,mode="rank"):
        self.replay_priority_queue = deque(maxlen=N)
        self.replay_index = [i for i in range(N)]
        self.weights = [1 for i in range(N)]
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.gamma = gamma
    def store(self):
        #print("store ..")
        if len(self.replay_priority_queue)==0:
            self.replay_priority_queue.append(1.0) 
        else:
            max_priority = max(self.replay_priority_queue)
            self.replay_priority_queue.append(max_priority) 
    def normalize(self,replay_priority):
        replay_priority = replay_priority ** self.alpha
        sum_priority = np.sum(replay_priority)
        replay_priority = replay_priority/sum_priority
        return replay_priority
    
    def rank_based_priority(self,replay_priority):
        replay_priority_index = np.argsort(replay_priority)[::-1]
        for i,index in enumerate(replay_priority_index):
            replay_priority[index] =  1.0 / (i+1.0)
        replay_priority = self.normalize(replay_priority)
        return replay_priority
        
    def sampling(self,replay_memory,batch_size):
        replay_priority = np.array(copy.copy(self.replay_priority_queue))
        replay_priority = replay_priority[:len(replay_memory)]
        if self.mode=="rank":
            replay_priority = self.rank_based_priority(replay_priority)
        replay_index = self.replay_index[:len(replay_priority)]            
        replay_batch_index = np.random.choice(replay_index,batch_size,p=replay_priority)
        try:
            replay_batch = deque([replay_memory[j] for j in replay_batch_index])
        except IndexError:
            print(replay_memory)
            print(len(replay_memory))
        
        N = len(replay_priority)
        
        for i in range(len(replay_priority)):
            self.weights[i] = (N*replay_priority[i])**(-self.beta)
        max_weights = max(self.weights)
        self.weights /= max_weights
        return replay_batch,replay_batch_index
    
    def update_priority(self,replay_batch_index,reward_batch,q_batch,next_q_batch):
        memo = []
        weights = []
        for i,index in enumerate(replay_batch_index):
            weights.append(self.weights[index])
            if index in memo:
                continue
            memo.append(index )
            with torch.no_grad():
                #print(self.gamma *next_q_batch[i] - q_batch[i])
                TD_error = float(reward_batch[i] + self.gamma *next_q_batch[i] - q_batch[i])
            self.replay_priority_queue[index] = abs(TD_error)
        weights  = np.array(weights,dtype=np.float64)
        #print(self.replay_priority_queue)
        return torch.from_numpy(weights)
        

class Multi_Step_Learning:
    def __init__(self,step_num=3,gamma=0.99):
        self.step_num = step_num
        self.gamma = gamma

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
    def arrange(self,episode):
        #The batch muste be [state,reward,next_state, done].
        for i in range(len(episode)):
            episode[i][1],episode[i][2] = self.__get_mult_step(episode,i,i+self.step_num)

        return episode
    def get_y_batch(self,done_batch,reward_batch, next_prediction_batch):
        return torch.cat(tuple(
                        reward if done[0] 
                        else self.gamma*reward + (self.gamma**self.step_num) * prediction for done ,reward, prediction 
                        in zip(done_batch,reward_batch, next_prediction_batch)))[:, None]     