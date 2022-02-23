
from collections import deque
import numpy as np
import copy

class PRIORITIZED_EXPERIENCE_REPLAY():
    def __init__(self,N ,alpha=0.7,beta=0.5,mode="rank"):
        self.replay_priority_queue = deque(maxlen=N)
        self.replay_index = [i for i in range(N)]
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
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
            replay_priority[index] = replay_priority[index]/ (i+1)
        replay_priority = self.normalize(replay_priority)
        return replay_priority
        
    def sampling(self,replay_memory,batch_size):
        replay_priority = np.array(copy.copy(self.replay_priority_queue))
        if self.mode=="rank":
            replay_priority = self.rank_based_priority(replay_priority)
        replay_index = self.replay_index[:len(replay_memory)]
        replay_batch_index = np.random.choice(replay_index,batch_size,p=replay_priority)
        replay_batch = deque([replay_memory[j] for j in replay_batch_index])
        
        return replay_batch,replay_batch_index
    
    def update_priority(self,y_batch,replay_batch_index):
        memo = []
        for j in replay_batch_index:
            if j in memo:
                continue
            memo.append(j)
            self.replay_priority_queue = abs(y_batch[j])
        exit()