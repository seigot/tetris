#!/usr/bin/python3
# -*- coding: utf-8 -*-

from tetris_model import BOARD_DATA, Shape
import math
from datetime import datetime
import numpy as np
import pprint

class TetrisAI(object):

    # Todo: collect followings to .json file...
    # use following data from board

    # shape.getCoords(direction, x0, 0): !!! how to get?
    # shape.shapeNone
    # BOARD_DATA.currentShape.getBoundingOffsets() ## get minx,max,miny...
    # BOARD_DATA.nextShape.getBoundingOffsets(0) !!! how to get?  ## get minx,max,miny...
    # BOARD_DATA.nextShape.getBoundingOffsets(d1) !!! how to get?
    # BOARD_DATA.nextShape.getCoords(d0, x0, 0): !!! how to get?
    # BOARD_DATA.getData()).reshape((BOARD_DATA.height, BOARD_DATA.width)) !!! how to get?
    # BOARD_DATA.currentShape, d0, x0) !!! how to get?
    
    board_data_width = 0
    board_data_height = 0
    ShapeNone_index = 0

    def nextMove(self, TetrisStatus):

        t1 = datetime.now()

        # print TetrisStatus
        print("=================================================>")
        pprint.pprint(TetrisStatus, width = 56, compact = True)

        # get data from TetrisStatus
        d0Range = TetrisStatus["shape"]["currentShape"]["direction_range"]
        d1Range = TetrisStatus["shape"]["nextShape"]["direction_range"]
        self.board_data_width = TetrisStatus["board"]["width"]
        self.board_data_height = TetrisStatus["board"]["height"]
        self.ShapeNone_index = TetrisStatus["debug_info"]["shape_info"]["shapeNone"]["index"]

        # search best strategy -->
        strategy = None
        LatestScore = 0
        for d0 in d0Range: # current shape direction range
            minX, maxX, _, _ = BOARD_DATA.currentShape.getBoundingOffsets(d0)
            for x0 in range(-minX, self.board_data_width - maxX): # x operation range
                board = self.calcStep1Board(d0, x0)
                for d1 in d1Range: # next shape direction range
                    minX, maxX, _, _ = BOARD_DATA.nextShape.getBoundingOffsets(d1)
                    dropDist = self.calcNextDropDist(board, d1, range(-minX, self.board_data_width - maxX))
                    for x1 in range(-minX, self.board_data_width - maxX):
                        score = self.calculateScore(np.copy(board), d1, x1, dropDist)
                        #if not strategy or strategy[2] < score:
                        #    strategy = (d0, x0, score)
                        if not strategy or LatestScore < score:
                            strategy = (d0, x0, 0)
                            LatestScore = score
        # search best strategy <--

        # return nextMove
        print("===", datetime.now() - t1)
        nextMove = {"strategy":
                      {
                        "x": "none",
                        "y_operation": "none",
                        "direction": "none",
                      },
                   }
        nextMove["strategy"]["x"] = strategy[1]
        nextMove["strategy"]["y_operation"] = strategy[2]
        nextMove["strategy"]["direction"] = strategy[0]
        return nextMove

    def calcNextDropDist(self, data, d0, xRange):
        res = {}
        for x0 in xRange:
            if x0 not in res:
                res[x0] = self.board_data_height - 1
            for x, y in BOARD_DATA.nextShape.getCoords(d0, x0, 0):
                yy = 0
                while yy + y < self.board_data_height and (yy + y < 0 or data[(y + yy), x] == self.ShapeNone_index):
                    yy += 1
                yy -= 1
                if yy < res[x0]:
                    res[x0] = yy
        return res

    def calcStep1Board(self, d0, x0):
        board = np.array(BOARD_DATA.getData()).reshape((self.board_data_height, self.board_data_width))
        self.dropDown(board, BOARD_DATA.currentShape, d0, x0)
        return board

    def dropDown(self, data, shape, direction, x0):
        dy = self.board_data_height - 1
        for x, y in shape.getCoords(direction, x0, 0):
            yy = 0
            while yy + y < self.board_data_height and (yy + y < 0 or data[(y + yy), x] == self.ShapeNone_index):
                yy += 1
            yy -= 1
            if yy < dy:
                dy = yy
        # print("dropDown: shape {0}, direction {1}, x0 {2}, dy {3}".format(shape.shape, direction, x0, dy))
        self.dropDownByDist(data, shape, direction, x0, dy)

    def dropDownByDist(self, data, shape, direction, x0, dist):
        for x, y in shape.getCoords(direction, x0, 0):
            data[y + dist, x] = shape.shape

    def calculateScore(self, step1Board, d1, x1, dropDist):
        # print("calculateScore")
        t1 = datetime.now()
        width = self.board_data_width
        height = self.board_data_height

        self.dropDownByDist(step1Board, BOARD_DATA.nextShape, d1, x1, dropDist[x1])
        # print(datetime.now() - t1)

        # Term 1: lines to be removed
        fullLines, nearFullLines = 0, 0
        roofY = [0] * width
        holeCandidates = [0] * width
        holeConfirm = [0] * width
        vHoles, vBlocks = 0, 0
        for y in range(height - 1, -1, -1):
            hasHole = False
            hasBlock = False
            for x in range(width):
                if step1Board[y, x] == self.ShapeNone_index:
                    hasHole = True
                    holeCandidates[x] += 1
                else:
                    hasBlock = True
                    roofY[x] = height - y
                    if holeCandidates[x] > 0:
                        holeConfirm[x] += holeCandidates[x]
                        holeCandidates[x] = 0
                    if holeConfirm[x] > 0:
                        vBlocks += 1
            if not hasBlock:
                break
            if not hasHole and hasBlock:
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

        score = fullLines * 1.8 - vHoles * 1.0 - vBlocks * 0.5 - maxHeight ** 1.5 * 0.02 \
            - stdY * 0.0 - stdDY * 0.01 - absDy * 0.2 - maxDy * 0.3
        # print(score, fullLines, vHoles, vBlocks, maxHeight, stdY, stdDY, absDy, roofY, d0, x0, d1, x1)
        return score


TETRIS_AI = TetrisAI()

