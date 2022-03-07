#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime
import pprint
import copy

import torch
import torch.nn as nn
from machine_learning.deep_q_network import DeepQNetwork
from collections import deque
#from tensorboardX import SummaryWriter

import os
import shutil
from random import random, randint, sample
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

    # GetNextMove is main function.
    # input
    #    nextMove : nextMove structure which is empty.
    #    GameStatus : block/field/judge/debug information. 
    #                 in detail see the internal GameStatus data.
    # output
    #    nextMove : nextMove structure which includes next shape position and the other.
    def GetNextMove(self, nextMove, GameStatus):

        t1 = datetime.now()

        # print GameStatus
        print("=================================================>")
        pprint.pprint(GameStatus, width = 61, compact = True)

        # get data from GameStatus
        # current shape info
        CurrentShapeDirectionRange = GameStatus["block_info"]["currentShape"]["direction_range"]
        self.CurrentShape_class = GameStatus["block_info"]["currentShape"]["class"]
        # next shape info
        NextShapeDirectionRange = GameStatus["block_info"]["nextShape"]["direction_range"]
        self.NextShape_class = GameStatus["block_info"]["nextShape"]["class"]
        # current board info
        self.board_backboard = GameStatus["field_info"]["backboard"]
        self.board_width = GameStatus["field_info"]["width"]
        self.board_height = GameStatus["field_info"]["height"]
        # default board definition
        self.board_data_width = GameStatus["field_info"]["width"]
        self.board_data_height = GameStatus["field_info"]["height"]
        self.ShapeNone_index = GameStatus["debug_info"]["shape_info"]["shapeNone"]["index"]
        self.mode = GameStatus["judge_info"]["mode"]
        strategy = (0, 0, 0, 0)

        # train/predict
        if self.mode == "train_sample":
            # train_sample -->

            # init parameter
            if self.init_train_parameter_flag == False:
                self.batch_size = 512
                self.lr = 1e-3
                self.gamma = 0.99
                self.initial_epsilon = 1
                self.final_epsilon = 1e-3
                self.num_decay_epochs = 1500
                self.num_epochs = 3000
                self.save_interval = 1000
                self.replay_memory_size = 30000
                self.log_path = "./tensorboard"
                self.saved_path = "./trained_models"

                self.episode = 0
                self.step = 0
                self.num_states = 4
                self.num_actions = 1
                self.model = DeepQNetwork()
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                self.criterion = nn.MSELoss()
                self.replay_memory = deque(maxlen=self.replay_memory_size)

                self.init_state_flag = True

                self.state = None
                self.next_state = None
                self.action = None
                self.reward = None

                self.king_of_score = 0

                if os.path.exists(self.log_path) == False:
                    os.makedirs(self.log_path)

                if os.path.exists(self.saved_path) == False:
                    os.makedirs(self.saved_path)

                #self.writer = SummaryWriter(self.log_path)
                self.init_train_parameter_flag = True

            # train main process -->
            done = False
            backboard = GameStatus["field_info"]["backboard"]

            print("### step ###")
            print(self.step)
            print("### episode ###")
            print(self.episode)

            if(self.init_state_flag == True):
                # init state
                fullLines_num, nHoles_num, nIsolatedBlocks_num, absDy_num = self.calcEvaluationValueSample(backboard)
                self.state = np.array([fullLines_num, nHoles_num, nIsolatedBlocks_num, absDy_num])
                self.state = torch.from_numpy(self.state).type(torch.FloatTensor)
                self.init_state_flag = False

            # get next actions
            next_actions, next_states = self.getStrategyAndStatelist(GameStatus)
            next_actions = np.array(next_actions)
            next_actions = torch.from_numpy(next_actions).type(torch.FloatTensor)
            next_states = np.array(next_states)
            next_states = torch.from_numpy(next_states).type(torch.FloatTensor)

            print("### next_actions ###")
            print(next_actions)

            print("### next_states ###")
            print(next_states)

            self.model.eval()
            with torch.no_grad():
                predictions = self.model(next_states)[:, 0]
                print("### predictions ###")
                print(predictions)

            epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.episode, 0) * (self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
            print("### epsilon ###")
            print(epsilon)
            u = random()
            random_action = u <= epsilon

            self.model.train()
            if random_action:
                print("### len(next states) ###")
                print(len(next_states))
                index = randint(0, len(next_states) - 1)
            else:
                index = torch.argmax(predictions).item()

            self.next_state = next_states[index, :]
            print("### self.next_state ###")
            print(self.next_state)
            self.action = next_actions[index]
            print("### self.action ###")
            print(self.action) # (rotation, position)

            # get nextMove
            direction = self.action[0].item()
            x = self.action[1].item()
            nextMove["strategy"]["direction"] = direction
            nextMove["strategy"]["x"] = x
            nextMove["strategy"]["y_operation"] = 1
            nextMove["strategy"]["y_moveblocknum"] = 0

            # get reward
            self.reward = 0
            trial_board = self.getBoard(backboard, self.CurrentShape_class, int(direction), int(x))
            fullLines_num, nHoles_num, nIsolatedBlocks_num, absDy_num = self.calcEvaluationValueSample(trial_board)
            removedlines = fullLines_num
            ## check if NextShape is appearable
            is_Continue = self.CheckIfContinue(trial_board, self.board_width, self.board_height, self.NextShape_class)

            if is_Continue == False:
                # gameover
                self.reward = torch.FloatTensor([-500])
                done = True

            elif self.step >= 180:
                # step max
                self.reward = torch.FloatTensor([10.0])
                done = True

            elif removedlines > 0:
                # get score
                if removedlines == 1:
                    linescore = 100 #Game_Manager.LINE_SCORE_1
                elif removedlines == 2:
                    linescore = 300 #Game_Manager.LINE_SCORE_2
                elif removedlines == 3:
                    linescore = 700 #Game_Manager.LINE_SCORE_3
                elif removedlines == 4:
                    linescore = 1300 #Game_Manager.LINE_SCORE_4
                self.reward = torch.FloatTensor([linescore])

            print("### state memory appned ###")
            print(self.state)
            self.replay_memory.append([self.state, self.reward, self.next_state, done])

            if done == True:
                # reset episode
                print("reset episode")

                self.episode += 1
                self.step = 0
                nextMove["option"]["reset_all_field"] = True

                final_score = GameStatus["judge_info"]["score"]
                final_tetrominoes = self.step
                final_cleared_lines = GameStatus["judge_info"]["line"]

                self.init_state_flag = True

                # update
                if len(self.replay_memory) < self.replay_memory_size / 10:
                    # skip update, because not enough data.
                    pass
                else:
                    # update model
                    batch = sample(self.replay_memory, min(len(self.replay_memory), self.batch_size))
                    state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
                    state_batch = torch.stack(tuple(self.state for self.state in state_batch))
                    reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
                    next_state_batch = torch.stack(tuple(self.state for self.state in next_state_batch))

                    q_values = self.model(state_batch)
                    print("### q_values ###")
                    print(q_values)
                    self.model.eval()
                    with torch.no_grad():
                        next_prediction_batch = self.model(next_state_batch)
                        print("### next prediction batch ###")
                        print(next_prediction_batch)
                    self.model.train()

                    y_batch = torch.cat(
                        tuple(self.reward if done else self.reward + self.gamma * prediction for self.reward, done, prediction in
                            zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

                    print("### y_batch ###")
                    print(y_batch)

                    self.optimizer.zero_grad()
                    loss = self.criterion(q_values, y_batch)
                    print("### loss ###")
                    print(loss)
                    loss.backward()
                    self.optimizer.step()

                    print("Episode: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
                        self.episode,
                        self.num_epochs,
                        self.action,
                        final_score,
                        final_tetrominoes,
                        final_cleared_lines))
                    #self.writer.add_scalar('Train/Score', final_score, self.episode - 1)
                    #self.writer.add_scalar('Train/Tetrominoes', final_tetrominoes, self.episode - 1)
                    #self.writer.add_scalar('Train/Cleared lines', final_cleared_lines, self.episode - 1)

                    if self.episode > 0 and self.episode % self.save_interval == 0:
                        torch.save(self.model, "{}/tetris_{}".format(self.saved_path, self.episode))

                    if final_score > self.king_of_score:
                        torch.save(self.model, "{}/tetris_{}_{}_{}".format(self.saved_path, self.episode, self.step, final_score))
                        torch.save(self.model, "{}/tetris".format(self.saved_path))
                        self.king_of_score = final_score
            else:
                self.state = self.next_state
 
            # step count up
            self.step += 1

            # train main process <--

        elif self.mode == "predict_sample":
            # predict_sample -->

            # init parameter
            if self.init_predict_parameter_flag == False:
                self.saved_path = "./trained_models"
                self.model = torch.load("{}/tetris".format(self.saved_path), map_location=lambda storage, loc: storage)
                self.model.eval()
                self.init_predict_parameter_flag = True

            # predict main process -->
            next_actions, next_states = self.getStrategyAndStatelist(GameStatus)
            next_actions = np.array(next_actions)
            next_actions = torch.from_numpy(next_actions).type(torch.FloatTensor)
            next_states = np.array(next_states)
            next_states = torch.from_numpy(next_states).type(torch.FloatTensor)

            predictions = self.model(next_states)[:, 0]
            index = torch.argmax(predictions).item()
            action = next_actions[index]

            print("===", datetime.now() - t1)
            nextMove["strategy"]["direction"] = action[0].item()
            nextMove["strategy"]["x"] = action[1].item()
            nextMove["strategy"]["y_operation"] = 1
            nextMove["strategy"]["y_moveblocknum"] = 0
            # predict main process <--

        print(nextMove)
        print("###### BLOCK_CONTROLLER_TRAIN_SAMPLE (mode:{}) ######".format(self.mode))
        return nextMove

    def getStrategyAndStatelist(self, GameStatus):

        # get data from GameStatus
        CurrentShapeDirectionRange = GameStatus["block_info"]["currentShape"]["direction_range"]
        CurrentShape_class = GameStatus["block_info"]["currentShape"]["class"]
        board_backboard = GameStatus["field_info"]["backboard"]

        # search best nextMove -->
        strategy = None
        strategy_list = []
        state_list = []
        # search with current block Shape
        for direction0 in CurrentShapeDirectionRange:
            # search with x range
            x0Min, x0Max = self.getSearchXRange(CurrentShape_class, direction0)
            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                board = self.getBoard(board_backboard, CurrentShape_class, direction0, x0)

                strategy = [direction0, x0]
                strategy_list.append(strategy)
                fullLines_num, nHoles_num, nIsolatedBlocks_num, absDy_num = self.calcEvaluationValueSample(board)
                state_list.append([fullLines_num, nHoles_num, nIsolatedBlocks_num, absDy_num])
        return strategy_list, state_list

    def CheckIfContinue(self, board, width, height, shape):
        # get Shape offsets
        _, _, minY, _ = shape.getBoundingOffsets(0)
        return self.tryMove(board, width, height, shape, 0, 5, -minY)

    def tryMove(self, board, width, height, shape, direction, x, y):
        # check Shape coordinates
        for x, y in shape.getCoords(direction, x, y):
            print(x, y)
            if x >= width or x < 0 or y >= height or y < 0:
                # out of range
                return False
            if board[x + y * width] > 0:
                # already block exist
                return False
        return True

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

    def calcEvaluationValueSample(self, board):
        #
        # sample function of evaluate board.
        #
        width = self.board_data_width
        height = self.board_data_height

        # evaluation paramters
        ## lines to be removed
        fullLines = 0
        ## number of holes or blocks in the line.
        nHoles, nIsolatedBlocks = 0, 0
        ## absolute differencial value of MaxY
        absDy = 0
        ## how blocks are accumlated
        BlockMaxY = [0] * width
        holeCandidates = [0] * width
        holeConfirm = [0] * width

        ### check board
        # each y line
        for y in range(height - 1, 0, -1):
            hasHole = False
            hasBlock = False
            # each x line
            for x in range(width):
                ## check if hole or block..
                if board[y * self.board_data_width + x] == self.ShapeNone_index:
                    # hole
                    hasHole = True
                    holeCandidates[x] += 1  # just candidates in each column..
                else:
                    # block
                    hasBlock = True
                    BlockMaxY[x] = height - y                # update blockMaxY
                    if holeCandidates[x] > 0:
                        holeConfirm[x] += holeCandidates[x]  # update number of holes in target column..
                        holeCandidates[x] = 0                # reset
                    if holeConfirm[x] > 0:
                        nIsolatedBlocks += 1                 # update number of isolated blocks

            if hasBlock == True and hasHole == False:
                # filled with block
                fullLines += 1
            elif hasBlock == True and hasHole == True:
                # do nothing
                pass
            elif hasBlock == False:
                # no block line (and ofcourse no hole)
                pass

        # nHoles
        for x in holeConfirm:
            nHoles += abs(x)

        ### absolute differencial value of MaxY
        BlockMaxDy = []
        for i in range(len(BlockMaxY) - 1):
            val = BlockMaxY[i] - BlockMaxY[i+1]
            BlockMaxDy += [val]
        for x in BlockMaxDy:
            absDy += abs(x)

        #### maxDy
        #maxDy = max(BlockMaxY) - min(BlockMaxY)
        #### maxHeight
        #maxHeight = max(BlockMaxY) - fullLines

        ## statistical data
        #### stdY
        #if len(BlockMaxY) <= 0:
        #    stdY = 0
        #else:
        #    stdY = math.sqrt(sum([y ** 2 for y in BlockMaxY]) / len(BlockMaxY) - (sum(BlockMaxY) / len(BlockMaxY)) ** 2)
        #### stdDY
        #if len(BlockMaxDy) <= 0:
        #    stdDY = 0
        #else:
        #    stdDY = math.sqrt(sum([y ** 2 for y in BlockMaxDy]) / len(BlockMaxDy) - (sum(BlockMaxDy) / len(BlockMaxDy)) ** 2)


        # calc Evaluation Value
        # score = 0
        # score = score + fullLines * 10.0           # try to delete line 
        # score = score - nHoles * 1.0               # try not to make hole
        # score = score - nIsolatedBlocks * 1.0      # try not to make isolated block
        # score = score - absDy * 1.0                # try to put block smoothly
        #score = score - maxDy * 0.3                # maxDy
        #score = score - maxHeight * 5              # maxHeight
        #score = score - stdY * 1.0                 # statistical data
        #score = score - stdDY * 0.01               # statistical data

        # print(score, fullLines, nHoles, nIsolatedBlocks, maxHeight, stdY, stdDY, absDy, BlockMaxY)
        #return score
        return fullLines, nHoles, nIsolatedBlocks, absDy

BLOCK_CONTROLLER_TRAIN_SAMPLE = Block_Controller()
