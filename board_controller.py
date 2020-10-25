#!/usr/bin/python3
# -*- coding: utf-8 -*-

import math
from datetime import datetime
import numpy as np
import pprint

class Board_Controller(object):

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
    def GetNextMove(self, GameStatus):

        t1 = datetime.now()

        # print GameStatus
        print("=================================================>")
        pprint.pprint(GameStatus, width = 56, compact = True)

        # get data from GameStatus
        CurrentShapeDirectionRange = GameStatus["block_info"]["currentShape"]["direction_range"]
        NextShapeDirectionRange = GameStatus["block_info"]["nextShape"]["direction_range"]
        self.board_backboard = GameStatus["field_info"]["backboard"]
        self.CurrentShape_class = GameStatus["block_info"]["currentShape"]["class"]
        self.NextShape_class = GameStatus["block_info"]["nextShape"]["class"]

        # search best nextMove -->
        # random sample
        direction      = np.random.randint(0,4)
        x              = np.random.randint(0,9)
        y_operation    = 0
        y_moveblocknum = np.random.randint(1,8)
        strategy = (direction, x, y_operation, y_moveblocknum)
        # search best nextMove <--

        # return nextMove
        print("===", datetime.now() - t1)
        nextMove = {"strategy":
                      {
                        "direction": "none",    # next shape direction ( 0 - 3 )
                        "x": "none",            # next x position (range: 0 - (witdh-1) )
                        "y_operation": "none",  # movedown or dropdown (0:movedown, 1:dropdown)
                        "y_moveblocknum": "none", # amount of next y movement
                      },
                   }
        nextMove["strategy"]["direction"] = strategy[0]
        nextMove["strategy"]["x"] = strategy[1]
        nextMove["strategy"]["y_operation"] = strategy[2]
        nextMove["strategy"]["y_moveblocknum"] = strategy[3]
        print(nextMove)
        return nextMove

 BOARD_CONTROLLER = Board_Controller()

