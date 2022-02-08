#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime
import pprint
import random

import copy
import torch
import torch.nn as nn
from model.deepqnet import DeepQNetwork,DeepQNetwork_v2

import omegaconf
from hydra import compose, initialize

import os
from tensorboardX import SummaryWriter
from collections import deque
from random import random, sample,randint
import numpy as np

class Block_Controller(object):

    # init parameter
    board_backboard = 0
    board_data_width = 0
    board_data_height = 0
    ShapeNone_index = 0
    CurrentShape_class = 0
    NextShape_class = 0

    def __init__(self):
        # init parameter
        self.mode = None
        # train
        self.init_train_parameter_flag = False
        # predict
        self.init_predict_parameter_flag = False
    
    def set_parameter(self):
        cfg = self.yaml_read()

        os.makedirs(cfg.common.dir,exist_ok=True)
        self.saved_path = cfg.common.dir + "/" + cfg.common.weight_path
        os.makedirs(self.saved_path ,exist_ok=True)
        self.writer = SummaryWriter(cfg.common.dir+"/"+cfg.common.log_path)

        self.log = cfg.common.dir+"/log.txt"
        self.log_score = cfg.common.dir+"/score.txt"
        self.log_reward = cfg.common.dir+"/reward.txt"

        self.state_dim = cfg.state.dim

        with open(self.log,"w") as f:
            print("start...", file=f)

        with open(self.log_score,"w") as f:
            print(0, file=f)

        with open(self.log_reward,"w") as f:
            print(0, file=f)

        print("model name: %s"%(cfg.model.name))
        if cfg.model.name=="DQN":
            self.model = DeepQNetwork(self.state_dim)
            self.initial_state = torch.FloatTensor([0 for i in range(self.state_dim)])
            self.get_next_func = self.get_next_states
            self.reshape_board = False
        elif cfg.model.name=="DQNv2":
            self.model = DeepQNetwork_v2()
            self.initial_state = torch.FloatTensor([[[0 for i in range(10)] for j in range(22)]])
            self.get_next_func = self.get_next_states_v2
            self.reshape_board = True
        self.lr = cfg.train.lr
        if cfg.train.optimizer=="Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()
            state = state.cuda()

        self.load_weight = cfg.common.load_weight

        self.replay_memory_size = cfg.train.replay_memory_size
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.criterion = nn.MSELoss()

        self.initial_epsilon = cfg.train.initial_epsilon
        self.final_epsilon = cfg.train.final_epsilon
        self.num_decay_epochs = cfg.train.num_decay_epochs


        self.num_epochs = cfg.train.num_epoch
        self.save_interval = cfg.train.save_interval
        self.gamma = cfg.train.gamma
        self.batch_size = cfg.train.batch_size

        self.height = 22
        self.width = 10
        self.epoch = 0
        self.score = 0
        self.max_score = -99999
        self.epoch_reward = 0
        self.cleared_lines = 0
        self.iter = 0
        
        
        self.state = self.initial_state 
        self.tetrominoes = 0
        self.max_tetrominoes = cfg.tetris.max_tetrominoes

        self.reward_clipping = cfg.train.reward_clipping

        self.score_list = cfg.tetris.score_list
        self.rewad_list = [0,self.score_list[1],self.score_list[2],self.score_list[3],self.score_list[4]]
        self.penalty = self.score_list[5]
        if self.reward_clipping:
            self.norm_num =max(max(self.rewad_list),abs(self.penalty))
            self.penalty /= self.norm_num
            self.rewad_list =[r/self.norm_num for r in self.rewad_list]
            
    def update(self):
        if self.mode=="train":
            self.score += self.score_list[5]
            self.replay_memory[-1][1] = self.penalty
            if len(self.replay_memory) < self.replay_memory_size / 10:
                print("================pass================")
                print("iter: {} ,meory: {}/{} , score: {}, clear line: {}, block: {} ".format(self.iter,
                len(self.replay_memory),self.replay_memory_size / 10,self.score,self.cleared_lines
                ,self.tetrominoes ))
            else:
                print("================update================")
                self.epoch += 1
                batch = sample(self.replay_memory, min(len(self.replay_memory),self.batch_size))
                state_batch, reward_batch, next_state_batch = zip(*batch)
                
                state_batch = torch.stack(tuple(state for state in state_batch))
                reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
                next_state_batch = torch.stack(tuple(state for state in next_state_batch))
                q_values = self.model(state_batch)
                self.model.eval()
                with torch.no_grad():
                    next_prediction_batch = self.model(next_state_batch)

                self.model.train()
                y_batch = torch.cat(
                    tuple(reward if reward<0 else reward + self.gamma * prediction for reward, prediction in
                          zip(reward_batch, next_prediction_batch)))[:, None]

                self.optimizer.zero_grad()
                loss = self.criterion(q_values, y_batch)
                loss.backward()
                self.optimizer.step()
                log = "Epoch: {} / {}, Score: {},  block: {},  Cleared lines: {}".format(
                    self.epoch,
                    self.num_epochs,
                    self.score,
                    self.tetrominoes,
                    self.cleared_lines
                    )
                print(log)
                with open(self.log,"a") as f:
                    print(log, file=f)
                with open(self.log_score,"a") as f:
                    print(self.score, file=f)

                with open(self.log_reward,"a") as f:
                    print(self.epoch_reward, file=f)
            if self.epoch > self.num_epochs:
                with open(self.log,"a") as f:
                    print("finish..", file=f)
                exit()
        else:
            self.epoch += 1
            log = "Epoch: {} / {}, Score: {},  block: {}, Reward: {}  Cleared lines: {}".format(
            self.epoch,
            self.num_epochs,
            self.score,
            self.tetrominoes,
            self.epoch_reward,
            self.cleared_lines
            )
            pass
    def yaml_read(self):
        #cfg = omegaconf.OmegaConf.load("config/default.yaml")
        initialize(config_path="../../config", job_name="tetris")
        cfg = compose(config_name="default")
        return cfg

    def reset_state(self):
            if self.score > self.max_score:
                torch.save(self.model, "{}/tetris_epoch_{}_score{}".format(self.saved_path,self.epoch,self.score))
                self.max_score  =  self.score

            self.state = self.initial_state
            self.score = 0
            self.cleared_lines = 0
            self.epoch_reward = 0
            self.tetrominoes = 0

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
        #if lines > 0:
        #    self.backBoard = newBackBoard
        return lines,board_new

    def get_bumpiness_and_height(self,board):
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    #各列の穴の個数を数える
    def get_holes(self, board):
        num_holes = 0
        for i in range(self.width):
            col = board[:,i]
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_state_properties(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    def get_state_properties_v2(self, board):
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        max_row = self.get_max_height(board)
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height,max_row])


    def get_max_height(self, board):
        sum_ = np.sum(board,axis=1)
        row = 0
        while row < self.height and sum_[row] ==0:
            row += 1
        return self.height - row

    def get_next_states_v2(self,GameStatus):
        states = {}
        piece_id =GameStatus["block_info"]["currentShape"]["index"]
        next_piece_id =GameStatus["block_info"]["nextShape"]["index"]
        #curr_piece = [row[:] for row in self.piece]
        if piece_id == 5:  # O piece
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7:
            num_rotations = 2
        else:
            num_rotations = 4
        CurrentShapeDirectionRange = GameStatus["block_info"]["currentShape"]["direction_range"]

        for direction0 in range(num_rotations):
            x0Min, x0Max = self.getSearchXRange(self.CurrentShape_class, direction0)
            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                board = self.getBoard(self.board_backboard, self.CurrentShape_class, direction0, x0)
                board = self.get_reshape_backboard(board)
                states[(x0, direction0)] = torch.from_numpy(board[np.newaxis,:,:]).float()
        return states

    def get_next_states(self,GameStatus):
        states = {}
        piece_id =GameStatus["block_info"]["currentShape"]["index"]
        next_piece_id =GameStatus["block_info"]["nextShape"]["index"]
        #curr_piece = [row[:] for row in self.piece]
        if piece_id == 5:  # O piece
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7:
            num_rotations = 2
        else:
            num_rotations = 4
        CurrentShapeDirectionRange = GameStatus["block_info"]["currentShape"]["direction_range"]

        for direction0 in range(num_rotations):
            x0Min, x0Max = self.getSearchXRange(self.CurrentShape_class, direction0)
            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                board = self.getBoard(self.board_backboard, self.CurrentShape_class, direction0, x0)
                board = self.get_reshape_backboard(board)
                if self.state_dim==5:
                   states[(x0, direction0)] = self.get_state_properties_v2(board)
                else:
                   states[(x0, direction0)] = self.get_state_properties(board)
        return states



            #curr_piece = self.rotate(curr_piece)
    def get_reshape_backboard(self,board):
        board = np.array(board)
        reshape_board = board.reshape(self.height,self.width)
        reshape_board = np.where(reshape_board>0,1,0)
        return reshape_board

    def step(self, action):
        x0, direction0 = action
        board = self.getBoard(self.board_backboard, self.CurrentShape_class, direction0, x0)

        board = self.get_reshape_backboard(board)

        #board[-1] = [1 for i in range(self.width)]
        lines_cleared, board = self.check_cleared_rows(board)
        #print(lines_cleared)
        #input()
        #score = 1 + (lines_cleared ** 2) * self.width
        reward = self.rewad_list[lines_cleared]
        self.epoch_reward += reward
        self.score += self.score_list[lines_cleared]
        self.cleared_lines += lines_cleared
        self.tetrominoes += 1
        return reward
           
    def GetNextMove(self, nextMove, GameStatus):

        t1 = datetime.now()
        self.mode = GameStatus["judge_info"]["mode"]
        if self.init_train_parameter_flag == False:
            self.init_train_parameter_flag = True
            self.set_parameter()
            
        # print GameStatus
        #print("=================================================>")
        #pprint.pprint(GameStatus, width = 61, compact = True)
        self.ind =GameStatus["block_info"]["currentShape"]["index"]
        self.board_backboard = GameStatus["field_info"]["backboard"]
        # default board definition
        self.board_data_width = GameStatus["field_info"]["width"]
        self.board_data_height = GameStatus["field_info"]["height"]
        self.CurrentShape_class = GameStatus["block_info"]["currentShape"]["class"]
        CurrentShapeDirectionRange = GameStatus["block_info"]["currentShape"]["direction_range"]
        self.CurrentShape_class = GameStatus["block_info"]["currentShape"]["class"]
        # next shape info
        NextShapeDirectionRange = GameStatus["block_info"]["nextShape"]["direction_range"]
        self.NextShape_class = GameStatus["block_info"]["nextShape"]["class"]
        self.ShapeNone_index = GameStatus["debug_info"]["shape_info"]["shapeNone"]["index"]

        reshape_backboard = self.get_reshape_backboard(GameStatus["field_info"]["backboard"])
        #self.state = reshape_backboard
        if self.reshape_board:
            self.state = torch.from_numpy(reshape_backboard[np.newaxis,:,:]).float()
        #self.model(data)   
        #exit()  
        # get data from GameStatus
        
        #next_steps = self.get_next_states(GameStatus)
        next_steps =self.get_next_func(GameStatus)
        if self.mode == "train":
            # init parameter
            epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.epoch, 0) * (
                    self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
            u = random()
            random_action = u <= epsilon
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
                       
            if torch.cuda.is_available():
                next_states = next_states.cuda()
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(next_states)[:, 0]

            self.model.train()
            if random_action:
                index = randint(0, len(next_steps) - 1)
            else:
                index = torch.argmax(predictions).item()
            next_state = next_states[index, :]
            action = next_actions[index]
            
            reward = self.step(action)
            self.replay_memory.append([self.state, reward, next_state])

            #print("===", datetime.now() - t1)
            nextMove["strategy"]["direction"] = action[1]
            nextMove["strategy"]["x"] = action[0]
            nextMove["strategy"]["y_operation"] = 1
            nextMove["strategy"]["y_moveblocknum"] = 1
            #print(nextMove)
            #print("###### SAMPLE CODE ######")
            self.state = next_state
            self.writer.add_scalar('Train/Score', self.score, self.epoch - 1)              

        elif self.mode == "predict":
            self.model.eval()
            next_actions, next_states = zip(*next_steps.items())
            next_states = torch.stack(next_states)
            predictions = self.model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action = next_actions[index]
            nextMove["strategy"]["direction"] = action[1]
            nextMove["strategy"]["x"] = action[0]
            nextMove["strategy"]["y_operation"] = 1
            nextMove["strategy"]["y_moveblocknum"] = 1

        return nextMove
    
    def getSearchXRange(self, Shape_class, direction):
        #
        # get x range from shape direction.
        #
        minX, maxX, _, _ = Shape_class.getBoundingOffsets(direction) # get shape x offsets[minX,maxX] as relative value.
        xMin = -1 * minX
        xMax = self.board_data_width - maxX
        return xMin, xMax

    def getShapeCoordArray(self, Shape_class, direction, x, y):
        #
        # get coordinate array by given shape.
        #
        coordArray = Shape_class.getCoords(direction, x, y) # get array from shape direction, x, y.
        return coordArray

    def getBoard(self, board_backboard, Shape_class, direction, x):
        # 
        # get new board.
        #
        # copy backboard data to make new board.
        # if not, original backboard data will be updated later.
        board = copy.deepcopy(board_backboard)
        _board = self.dropDown(board, Shape_class, direction, x)
        return _board

    def dropDown(self, board, Shape_class, direction, x):
        # 
        # internal function of getBoard.
        # -- drop down the shape on the board.
        # 
        dy = self.board_data_height - 1
        coordArray = self.getShapeCoordArray(Shape_class, direction, x, 0)
        # update dy
        for _x, _y in coordArray:
            _yy = 0
            while _yy + _y < self.board_data_height and (_yy + _y < 0 or board[(_y + _yy) * self.board_data_width + _x] == self.ShapeNone_index):
                _yy += 1
            _yy -= 1
            if _yy < dy:
                dy = _yy
        # get new board
        _board = self.dropDownWithDy(board, Shape_class, direction, x, dy)
        return _board

    def dropDownWithDy(self, board, Shape_class, direction, x, dy):
        #
        # internal function of dropDown.
        #
        _board = board
        coordArray = self.getShapeCoordArray(Shape_class, direction, x, 0)
        for _x, _y in coordArray:
            _board[(_y + dy) * self.board_data_width + _x] = Shape_class.shape
        return _board
BLOCK_CONTROLLER_TRAIN = Block_Controller()
