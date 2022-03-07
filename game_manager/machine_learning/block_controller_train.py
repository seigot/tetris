#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime
import pprint
import random

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
    #    GameStatus : this data include all field status, 
    #                 in detail see the internal GameStatus data.
    # output
    #    nextMove : this data include next shape position and the other,
    #               if return None, do nothing to nextMove.
    def GetNextMove(self, nextMove, GameStatus):

        t1 = datetime.now()

        # print GameStatus
        print("=================================================>")
        pprint.pprint(GameStatus, width = 61, compact = True)

        # get data from GameStatus
        self.mode = GameStatus["judge_info"]["mode"]

        if self.mode == "train":
            # init parameter
            if self.init_train_parameter_flag == False:
                self.init_train_parameter_flag = True
            # train main process -->
            # train main process <--

        elif self.mode == "predict":
            # init parameter
            if self.init_predict_parameter_flag == False:
                self.init_predict_parameter_flag = True
            # predict main process -->
            # predict main process <--

        # search best nextMove -->
        # random sample
        nextMove["strategy"]["direction"] = random.randint(0,4)
        nextMove["strategy"]["x"] = random.randint(0,9)
        nextMove["strategy"]["y_operation"] = 1
        nextMove["strategy"]["y_moveblocknum"] = random.randint(1,8)
        nextMove["option"]["reset_all_field"] = False
        # search best nextMove <--

        # return nextMove
        print("===", datetime.now() - t1)
        print(nextMove)
        print("###### BLOCK_CONTROLLER_TRAIN (mode:{}) ######".format(self.mode))
        return nextMove

BLOCK_CONTROLLER_TRAIN = Block_Controller()

