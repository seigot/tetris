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
        self.board_data_width = GameStatus["field_info"]["width"]
        self.board_data_height = GameStatus["field_info"]["height"]
        self.ShapeNone_index = GameStatus["debug_info"]["shape_info"]["shapeNone"]["index"]
        self.CurrentShape_class = GameStatus["block_info"]["currentShape"]["class"]
        self.NextShape_class = GameStatus["block_info"]["nextShape"]["class"]

        # search best nextMove -->
        strategy = None
        LatestScore = 0
        for d0 in CurrentShapeDirectionRange:
            # get CurrentShape X range when direction "d0"
            minX, maxX, _, _ = self.CurrentShape_class.getBoundingOffsets(d0)
            for x0 in range(-minX, self.board_data_width - maxX):
                # temporally drop and get board when direction "d0" and "x0"
                board = self.calcStep1Board(d0, x0)
                for d1 in NextShapeDirectionRange:
                    # get NextShape X range when direction "d1"
                    minX, maxX, _, _ = self.NextShape_class.getBoundingOffsets(d1)
                    # get dropDist to caluculate post process effectively
                    dropDist = self.calcNextDropDist(board, d1, range(-minX, self.board_data_width - maxX))
                    for x1 in range(-minX, self.board_data_width - maxX):
                        # calculate score with the conbination "d0,x0" and "d1,x1"
                        score = self.calculateScore(np.copy(board), d1, x1, dropDist)
                        if not strategy or LatestScore < score:
                            strategy = (d0, x0, 0)
                            LatestScore = score
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
        nextMove["strategy"]["y_operation"] = 0
        nextMove["strategy"]["y_moveblocknum"] = strategy[2]
        print(nextMove)
        # if play manually, return None
        # return None

        return nextMove

    def calcNextDropDist(self, data, d0, xRange):
        res = {}
        for x0 in xRange:
            if x0 not in res:
                res[x0] = self.board_data_height - 1
            for x, y in self.NextShape_class.getCoords(d0, x0, 0):
                yy = 0
                while yy + y < self.board_data_height and (yy + y < 0 or data[(y + yy), x] == self.ShapeNone_index):
                    yy += 1
                yy -= 1
                if yy < res[x0]:
                    res[x0] = yy
        return res

    def calcStep1Board(self, d0, x0):
        board = np.array(self.board_backboard).reshape((self.board_data_height, self.board_data_width))
        self.dropDown(board, self.CurrentShape_class, d0, x0)
        return board

    def dropDown(self, data, Shape_class, direction, x0):
        dy = self.board_data_height - 1
        for x, y in Shape_class.getCoords(direction, x0, 0):
            yy = 0
            while yy + y < self.board_data_height and (yy + y < 0 or data[(y + yy), x] == self.ShapeNone_index):
                yy += 1
            yy -= 1
            if yy < dy:
                dy = yy
        self.dropDownByDist(data, Shape_class, direction, x0, dy)

    def dropDownByDist(self, data, Shape_class, direction, x0, dist):
        for x, y in Shape_class.getCoords(direction, x0, 0):
            data[y + dist, x] = Shape_class.shape

    def calculateScore(self, step1Board, d1, x1, dropDist):
        t1 = datetime.now()
        width = self.board_data_width
        height = self.board_data_height

        # temporally drop and get board with direction "d1" and "x1"
        self.dropDownByDist(step1Board, self.NextShape_class, d1, x1, dropDist[x1])

        # Term 1: lines to be removed
        fullLines = 0
        roofY = [0] * width
        holeCandidates = [0] * width
        holeConfirm = [0] * width
        vHoles, vBlocks = 0, 0
        ## check all line on board
        for y in range(height - 1, -1, -1):
            hasHole = False
            hasBlock = False
            for x in range(width):
                ## check if hole exists
                if step1Board[y, x] == self.ShapeNone_index:
                    hasHole = True
                    holeCandidates[x] += 1 # just candidates
                else:
                    hasBlock = True
                    roofY[x] = height - y
                    if holeCandidates[x] > 0:
                        holeConfirm[x] += holeCandidates[x]
                        holeCandidates[x] = 0
                    if holeConfirm[x] > 0:
                        vBlocks += 1
            if not hasBlock:
                # no block line (and ofcourse no hole)
                break
            if not hasHole and hasBlock:
                # filled with block
                fullLines += 1
        vHoles = sum([x ** .7 for x in holeConfirm])
        maxHeight = max(roofY) - fullLines
        # print(datetime.now() - t1)

        roofDy = [roofY[i] - roofY[i+1] for i in range(len(roofY) - 1)]

        if len(roofY) <= 0:
            stdY = 0
        else:
            stdY = math.sqrt(sum([y ** 2 for y in roofY]) / len(roofY) - (sum(roofY) / len(roofY)) ** 2)
        if len(roofDy) <= 0:
            stdDY = 0
        else:
            stdDY = math.sqrt(sum([y ** 2 for y in roofDy]) / len(roofDy) - (sum(roofDy) / len(roofDy)) ** 2)

        absDy = sum([abs(x) for x in roofDy])
        maxDy = max(roofY) - min(roofY)
        # print(datetime.now() - t1)

        # score Evaluation
        score = 0
        score = score + fullLines * 1.8            # try     to delete line 
        score = score - vHoles * 1.0               # try not to make hole
        score = score - vBlocks * 0.5              # try not to set block
        score = score - maxHeight ** 1.5 * 0.02    # try     to make maxheight smaller
        score = score - stdY * 0.0                 # statistical data
        score = score - stdDY * 0.01               # statistical data
        score = score - absDy * 0.2                # statistical data
        score = score - maxDy * 0.3                # statistical data

        # print(score, fullLines, vHoles, vBlocks, maxHeight, stdY, stdDY, absDy, roofY, d0, x0, d1, x1)
        return score


BOARD_CONTROLLER = Board_Controller()

