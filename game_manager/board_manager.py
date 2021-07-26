#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np_randomShape
import numpy as np_randomObstacle
import numpy as np_randomObstaclePiece
import copy

# Shape manager
class Shape(object):
    shapeNone = 0
    shapeI = 1
    shapeL = 2
    shapeJ = 3
    shapeT = 4
    shapeO = 5
    shapeS = 6
    shapeZ = 7

    # shape1 : ****
    #          ----
    #          ----
    #
    # shape2 : -*--
    #          -*--
    #          -**-
    #
    # shape3 : --*-
    #          --*-
    #          -**-
    #
    # shape4 : -*--
    #          -**-
    #          -*--
    #
    # shape5 : -**-
    #          -**-
    #          ----
    #
    # shape6 : -**-
    #          **--
    #          ----
    #
    # shape7 : **--
    #          -**-
    #          ----
    #

    shapeCoord = (
        ((0, 0), (0, 0), (0, 0), (0, 0)),
        ((0, -1), (0, 0), (0, 1), (0, 2)),
        ((0, -1), (0, 0), (0, 1), (1, 1)),
        ((0, -1), (0, 0), (0, 1), (-1, 1)),
        ((0, -1), (0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, -1), (1, 0), (1, -1)),
        ((0, 0), (0, -1), (-1, 0), (1, -1)),
        ((0, 0), (0, -1), (1, 0), (-1, -1))
    )

    def __init__(self, shape=0):
        self.shape = shape

    def getRotatedOffsets(self, direction):
        tmpCoords = Shape.shapeCoord[self.shape]
        if direction == 0 or self.shape == Shape.shapeO:
            return ((x, y) for x, y in tmpCoords)

        if direction == 1:
            return ((-y, x) for x, y in tmpCoords)

        if direction == 2:
            if self.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
                return ((x, y) for x, y in tmpCoords)
            else:
                return ((-x, -y) for x, y in tmpCoords)

        if direction == 3:
            if self.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
                return ((-y, x) for x, y in tmpCoords)
            else:
                return ((y, -x) for x, y in tmpCoords)

    def getCoords(self, direction, x, y):
        return ((x + xx, y + yy) for xx, yy in self.getRotatedOffsets(direction))

    def getBoundingOffsets(self, direction):
        tmpCoords = self.getRotatedOffsets(direction)
        minX, maxX, minY, maxY = 0, 0, 0, 0
        for x, y in tmpCoords:
            if minX > x:
                minX = x
            if maxX < x:
                maxX = x
            if minY > y:
                minY = y
            if maxY < y:
                maxY = y
        return (minX, maxX, minY, maxY)


# board manager
class BoardData(object):

    width = 10
    height = 22

    def __init__(self):
        self.backBoard = [0] * BoardData.width * BoardData.height # initialize board matrix

        self.currentX = -1
        self.currentY = -1
        self.currentDirection = 0
        self.currentShape = Shape() # initial current shape data
        self.nextShape = None
        self.shape_info_stat = [0] * 8
        self.obstacle_height = 0
        self.obstacle_probability = 0
        self.random_seed = 0
        self.nextShapeIndexCnt = 1

    def init_randomseed(self, num):
        self.random_seed = int(num % (2**32-1))
        np_randomShape.random.seed(self.random_seed)
        np_randomObstacle.random.seed(self.random_seed)
        np_randomObstaclePiece.random.seed(self.random_seed)

    def init_obstacle_parameter(self, height, probability):
        self.obstacle_height = height
        self.obstacle_probability = probability

    def getData(self):
        return self.backBoard[:]

    def getDataWithCurrentBlock(self):
        tmp_backboard = copy.deepcopy(self.backBoard)
        Shape_class = self.currentShape
        direction = self.currentDirection
        x = self.currentX
        y = self.currentY
        coordArray = Shape_class.getCoords(direction, x, y)
        for _x, _y in coordArray:
            tmp_backboard[_y * self.width + _x] = Shape_class.shape
        return tmp_backboard[:]

    def getValue(self, x, y):
        return self.backBoard[x + y * BoardData.width]

    def getCurrentShapeCoord(self):
        return self.currentShape.getCoords(self.currentDirection, self.currentX, self.currentY)

    def getNewShapeIndex(self):
        if self.random_seed == 0:
            # static value
            nextShapeIndex = self.nextShapeIndexCnt
            self.nextShapeIndexCnt += 1
            if self.nextShapeIndexCnt >= (7+1):
                self.nextShapeIndexCnt = 1
        else:
            # random value
            nextShapeIndex = np_randomShape.random.randint(1, 8)
        return nextShapeIndex

    def createNewPiece(self):
        if self.nextShape == None:
            self.nextShape = Shape(self.getNewShapeIndex()) # initial next shape data

        minX, maxX, minY, maxY = self.nextShape.getBoundingOffsets(0)
        result = False
        if self.tryMoveCurrent(0, 5, -minY):
            self.currentX = 5
            self.currentY = -minY
            self.currentDirection = 0
            self.currentShape = self.nextShape
            self.nextShape = Shape(self.getNewShapeIndex())
            result = True
        else:
            self.currentShape = Shape()
            self.currentX = -1
            self.currentY = -1
            self.currentDirection = 0
            result = False
        self.shape_info_stat[self.currentShape.shape] += 1
        return result

    def tryMoveCurrent(self, direction, x, y):
        return self.tryMove(self.currentShape, direction, x, y)

    def tryMove(self, shape, direction, x, y):
        for x, y in shape.getCoords(direction, x, y):
            if x >= BoardData.width or x < 0 or y >= BoardData.height or y < 0:
                return False
            if self.backBoard[x + y * BoardData.width] > 0:
                return False
        return True

    def moveDown(self):
        # move piece, 1 block
        # and return the number of lines which is removed in this function.
        removedlines = 0
        moveDownlines = 0
        if self.tryMoveCurrent(self.currentDirection, self.currentX, self.currentY + 1):
            self.currentY += 1
            moveDownlines += 1
        else:
            self.mergePiece()
            removedlines = self.removeFullLines()
            self.createNewPiece()
        return removedlines, moveDownlines

    def dropDown(self):
        # drop piece, immediately
        # and return the number of lines which is removed in this function.
        dropdownlines = 0
        while self.tryMoveCurrent(self.currentDirection, self.currentX, self.currentY + 1):
            self.currentY += 1
            dropdownlines += 1

        self.mergePiece()
        removedlines = self.removeFullLines()
        self.createNewPiece()
        return removedlines, dropdownlines

    def moveLeft(self):
        if self.tryMoveCurrent(self.currentDirection, self.currentX - 1, self.currentY):
            self.currentX -= 1
        else:
            print("failed to moveLeft..")
            return False
        return True

    def moveRight(self):
        if self.tryMoveCurrent(self.currentDirection, self.currentX + 1, self.currentY):
            self.currentX += 1
        else:
            print("failed to moveRight..")
            return False
        return True

    def rotateRight(self):
        if self.tryMoveCurrent((self.currentDirection + 1) % 4, self.currentX, self.currentY):
            self.currentDirection += 1
            self.currentDirection %= 4
        else:
            print("failed to rotateRight..")
            return False
        return True

    def rotateLeft(self):
        if self.tryMoveCurrent((self.currentDirection - 1) % 4, self.currentX, self.currentY):
            self.currentDirection -= 1
            self.currentDirection %= 4
        else:
            print("failed to rotateLeft..")
            return False
        return True

    def removeFullLines(self):
        newBackBoard = [0] * BoardData.width * BoardData.height
        newY = BoardData.height - 1
        lines = 0
        for y in range(BoardData.height - 1, -1, -1):
            blockCount = sum([1 if self.backBoard[x + y * BoardData.width] > 0 else 0 for x in range(BoardData.width)])
            if blockCount < BoardData.width:
                for x in range(BoardData.width):
                    newBackBoard[x + newY * BoardData.width] = self.backBoard[x + y * BoardData.width]
                newY -= 1
            else:
                lines += 1
        if lines > 0:
            self.backBoard = newBackBoard
        return lines

    def mergePiece(self):
        for x, y in self.currentShape.getCoords(self.currentDirection, self.currentX, self.currentY):
            self.backBoard[x + y * BoardData.width] = self.currentShape.shape

        self.currentX = -1
        self.currentY = -1
        self.currentDirection = 0
        self.currentShape = Shape()

    def clear(self):
        self.currentX = -1
        self.currentY = -1
        self.currentDirection = 0
        self.currentShape = Shape()
        self.backBoard = [0] * BoardData.width * BoardData.height
        self.addobstacle()

    def addobstacle(self):
        obstacle_height = self.obstacle_height
        obstacle_probability = self.obstacle_probability

        for y in range(BoardData.height):
            line_obstacle_cnt = 0
            for x in range(BoardData.width):

                if y < (BoardData.height - obstacle_height):
                    continue
                if line_obstacle_cnt >= (BoardData.width - 1):
                    continue

                # create obstacle
                tmp_num = np_randomObstacle.random.randint(1, 100)
                if tmp_num <= obstacle_probability:
                    self.backBoard[x + y * BoardData.width] = np_randomObstaclePiece.random.randint(1, 8)
                    line_obstacle_cnt += 1

BOARD_DATA = BoardData()
