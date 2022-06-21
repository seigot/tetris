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

    # GetNextMove is main function.
    # input
    #    GameStatus : this data include all field status, 
    #                 in detail see the internal GameStatus data.
    # output
    #    nextMove : this data include next shape position and the other,
    #               if return None, do nothing to nextMove.
    def GetNextMove(self, nextMove, GameStatus,yaml_file=None,weight=None):

        t1 = datetime.now()

        # print GameStatus
        print("=================================================>")
        print("block_controller_sample2")
        pprint.pprint(GameStatus, width = 61, compact = True)

        # search best nextMove -->
        # random sample
        nextMove["strategy"]["direction"] = random.randint(0,4)
        nextMove["strategy"]["x"] = random.randint(0,9)
        nextMove["strategy"]["y_operation"] = 1
        nextMove["strategy"]["y_moveblocknum"] = random.randint(1,8)
        nextMove["option"]["reset_callback_function_addr"] = self.update
        # search best nextMove <--

        # return nextMove
        print("===", datetime.now() - t1)
        print(nextMove)
        return nextMove

    def update(self):
        print("update")

BLOCK_CONTROLLER_TRAIN_SAMPLE2 = Block_Controller()

