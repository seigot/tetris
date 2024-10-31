import pprint
import random

#!/usr/bin/python3
class Block_Controller(object):
# -*- coding: utf-8 -*-

from datetime import datetime
import pprint
import random
class Block_Controller(object):

    def __init__(self):
        self.block_sequence = self.generate_block_sequence()
        self.current_block_index = 0


    def generate_block_sequence(self):
class Block_Controller(object):
        # Generate a fixed sequence of 180 blocks for level 1

        sequence = []
    # init parameter
        for _ in range(25):  # 25 full cycles of 7 blocks
    board_backboard = 0
            sequence.extend([1, 2, 3, 4, 5, 6, 7])
    board_data_width = 0
        sequence.extend([1, 2, 3, 4, 5])  # Add 5 more to make 180
    board_data_height = 0
        return sequence
    ShapeNone_index = 0
    def GetNextMove(self, nextMove, GameStatus):
    CurrentShape_class = 0
        # Use the pre-generated block sequence
    NextShape_class = 0
        next_block = self.block_sequence[self.current_block_index]

        self.current_block_index += 1
    # GetNextMove is main function.
        # Implement logic to determine the move for the next block
    # input
        # This is a placeholder for the actual move logic
    #    GameStatus : this data include all field status, 
        nextMove['rotate'] = 0
    #                 in detail see the internal GameStatus data.
        nextMove['move'] = 0
    # output
        nextMove['drop'] = 1
    #    nextMove : this data include next shape position and the other,
        return nextMove
    #               if return None, do nothing to nextMove.
    def GetNextMove(self, nextMove, GameStatus):

        t1 = datetime.now()

        # print GameStatus
        print("=================================================>")
        pprint.pprint(GameStatus, width = 61, compact = True)

        # search best nextMove -->
        # random sample
        nextMove["strategy"]["direction"] = random.randint(0,4)
        nextMove["strategy"]["x"] = random.randint(0,9)
        nextMove["strategy"]["y_operation"] = 1
        nextMove["strategy"]["y_moveblocknum"] = random.randint(1,8)
        # search best nextMove <--

        # return nextMove
        print("===", datetime.now() - t1)
        print(nextMove)
        return nextMove

BLOCK_CONTROLLER = Block_Controller()

