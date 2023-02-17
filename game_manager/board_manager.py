#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np_randomShape
import numpy as np_randomObstacle
import numpy as np_randomObstaclePiece
import copy

#####################################
#####################################
# テトリミノ形状
# Shape manager
#####################################
#####################################
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
    #テトリミノ形状座標タプル
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

    ##############################
    # テトリミノ形状を回転した座標を返す
    # direction: テトリミノ回転方向
    ##############################
    def getRotatedOffsets(self, direction):
        # テトリミノ形状座標タプルを取得
        tmpCoords = Shape.shapeCoord[self.shape]
        # 方向によってテトリミノ形状座標タプルを回転させる
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

    ###################
    # direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置した場合の座標配列を返す
    ###################
    def getCoords(self, direction, x, y):
        return ((x + xx, y + yy) for xx, yy in self.getRotatedOffsets(direction))

    ###################
    # テトリミノが原点から x,y 両方向に最大何マス占有するのか返す
    ###################
    def getBoundingOffsets(self, direction):
        # テトリミノ形状を回転した座標を返す
        tmpCoords = self.getRotatedOffsets(direction)
        # 
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


#####################################################################
#####################################################################
# board manager
#####################################################################
#####################################################################
class BoardData(object):

    width = 10
    height = 22

    #######################################
    ##  board manager 初期化
    #######################################
    def __init__(self):
        self.backBoard = [0] * BoardData.width * BoardData.height # initialize board matrix

        self.currentX = -1
        self.currentY = -1
        self.currentDirection = 0
        self.currentShape = Shape() # initial current shape data
        self.nextShape = None
        self.holdShape = None
        self.shape_info_stat = [0] * 8
        self.obstacle_height = 0
        self.obstacle_probability = 0
        self.random_seed = 0
        self.nextShapeIndexCnt = 0
        self.nextShapeIndexList = [1,2,3,4,5,6,7]
        self.nextShapeIndexListDXY = [[0,0,1] for _ in range(len(self.nextShapeIndexList))] # for art DXY config data
        self.colorTable = [0x000000, 0xCC6666, 0x66CC66, 0x6666CC,
                           0xCCCC66, 0xCC66CC, 0x66CCCC, 0xDAAA00]
        self.tryMoveNextCnt = 0
        self.ShapeListMax = 2
        # ShapeList
        #  ShapeNumber 0: currentShape
        #  ShapeNumber 1: nextShape
        #  ShapeNumber 2: next nextShape
        #  ...
        self.ShapeList = []

    #######################################
    ## 乱数初期化
    #######################################
    def init_randomseed(self, num):
        self.random_seed = int(num % (2**32-1))
        np_randomShape.random.seed(self.random_seed)
        np_randomObstacle.random.seed(self.random_seed)
        np_randomObstaclePiece.random.seed(self.random_seed)

    #######################################
    ## テトリミノ形状初期化
    #######################################
    def init_shape_parameter(self, ShapeListMax):
        self.ShapeListMax = ShapeListMax

    #######################################
    ## 障害ブロック初期化
    #######################################
    def init_obstacle_parameter(self, height, probability):
        self.obstacle_height = height
        self.obstacle_probability = probability

    #######################################
    ## art用configを利用してcolorTable/nextShapeIndexListを初期化
    #######################################
    def init_art_config(self, art_config):
        # open colorTable, nextShapeIndexList config file for art
        try:
            import json
            art_config_filepath = art_config
            with open(art_config_filepath, 'r') as json_open:
                json_load = json.load(json_open)
            self.colorTable[1] = int(json_load['color']['shapeI'],16)
            self.colorTable[2] = int(json_load['color']['shapeL'],16)
            self.colorTable[3] = int(json_load['color']['shapeJ'],16)
            self.colorTable[4] = int(json_load['color']['shapeT'],16)
            self.colorTable[5] = int(json_load['color']['shapeO'],16)
            self.colorTable[6] = int(json_load['color']['shapeS'],16)
            self.colorTable[7] = int(json_load['color']['shapeZ'],16)
            block_order = list(json_load['block_order'])
            self.nextShapeIndexList = [ block_order[ii][0] for ii in range(len(block_order))]
            self.nextShapeIndexListDXY = [[block_order[ii][1],block_order[ii][2],block_order[ii][3]] for ii in range(len(block_order))]
        except Exception as e:
            #print(e)
            pass

    #######################################
    ## 画面ボードデータを返す
    #######################################
    def getData(self):
        return self.backBoard[:]

    #######################################
    ## 画面ボードデータコピーし動いているテトリミノデータを付加
    #######################################
    def getDataWithCurrentBlock(self):
        # 動いているテトリミノ以外の画面ボード状態取得
        tmp_backboard = copy.deepcopy(self.backBoard)
        # テトリミノの現状形状取得
        Shape_class = self.currentShape
        # テトリミノの回転状態取得
        direction = self.currentDirection
        # 現状のテトリミノ中心位置取得
        x = self.currentX
        y = self.currentY
        # x,y 座標に回転されたテトリミノを置いた場合の座標配列を取得する
        coordArray = Shape_class.getCoords(direction, x, y)
        # 上記配列に従って画面ボードに動いているテトリミノのデータを書き込んでいく
        for _x, _y in coordArray:
            tmp_backboard[_y * self.width + _x] = Shape_class.shape
        return tmp_backboard[:]

    #######################################
    ## 画面ボード上のテトリミノを返す
    #######################################
    def getValue(self, x, y):
        return self.backBoard[x + y * BoardData.width]

    #######################################
    ## 予告テトリミノ配列の長さを返す
    #######################################
    def getShapeListLength(self):
        length = len(self.ShapeList)
        return length

    ################################
    # テトリミノクラスデータ, テトリミノ座標配列、テトリミノ回転種類を返す
    # ShapeClass ... shape のオブジェクト
    ################################
    def getShapeDataFromShapeClass(self, ShapeClass):

        if ShapeClass == None:
            return None, None, None

        #ShapeClass = self.ShapeList[ShapeNumber]
        ShapeIdx = ShapeClass.shape
        ShapeRange = (0, 1, 2, 3)
        if ShapeIdx in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
            ShapeRange = (0, 1)
        elif ShapeIdx == Shape.shapeO:
            ShapeRange = (0,)
        else:
            ShapeRange = (0, 1, 2, 3)

        return ShapeClass, ShapeIdx, ShapeRange

    ################################
    # 予告テトリミノ配列番号から shape オブジェクトを返す
    # ShapeNumber ... 予告テトリミノ配列番号　
    ################################
    def getShapeData(self, ShapeNumber):
        ShapeClass = self.ShapeList[ShapeNumber]
        return self.getShapeDataFromShapeClass(ShapeClass)

    ################################
    # ホールド shape オブジェクトを返す
    ################################
    def getholdShapeData(self):
        return self.getShapeDataFromShapeClass(self.holdShape)

    ################################
    # getcolorTableを返す
    ################################
    def getcolorTable(self):
        return self.colorTable

    ################################
    # nextShapeIndexListDXYを返す
    ################################
    def getnextShapeIndexListDXY(self, index):
        index = index % len(self.nextShapeIndexListDXY)
        d = self.nextShapeIndexListDXY[index][0]
        x = self.nextShapeIndexListDXY[index][1]
        y = self.nextShapeIndexListDXY[index][2]
        return d,x,y

    #################################################
    # direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置した場合の座標配列を返す
    #################################################
    def getCurrentShapeCoord(self):
        return self.currentShape.getCoords(self.currentDirection, self.currentX, self.currentY)

    #################################################
    # 次のテトリミノの取得
    #################################################
    def getNewShapeIndex(self):
        if self.random_seed == 0:
            # static value
            nextShapeIndex = self.nextShapeIndexList[self.nextShapeIndexCnt]
            self.nextShapeIndexCnt += 1
            if self.nextShapeIndexCnt >= len(self.nextShapeIndexList):
                self.nextShapeIndexCnt = 0
        else:
            # random value
            nextShapeIndex = np_randomShape.random.randint(1, 8)
        return nextShapeIndex

    #####################################
    ## 新しい予告テトリミノ配列作成
    ####################################
    def createNewPiece(self):
        if self.nextShape == None:
            self.ShapeList.append(0)
            ## 次のテトリミノデータ作成
            # initialize next shape data
            for i in range(self.ShapeListMax-1):
                self.ShapeList.append(Shape(self.getNewShapeIndex()))
            self.nextShape = self.ShapeList[1]

        # テトリミノが原点から x,y 両方向に最大何マス占有するのか取得
        minX, maxX, minY, maxY = self.nextShape.getBoundingOffsets(0)
        result = False

        # check if nextShape can appear
        if self.tryMoveNext(0, 5, -minY):
            self.currentX = 5
            self.currentY = -minY
            self.currentDirection = 0
            # get nextShape
            self.ShapeList.pop(0)
            self.ShapeList.append(Shape(self.getNewShapeIndex()))
            self.currentShape = self.ShapeList[0]
            self.nextShape = self.ShapeList[1]
            result = True
        else:
            # cannnot appear
            self.currentShape = Shape()
            self.currentX = -1
            self.currentY = -1
            self.currentDirection = 0
            result = False
        self.shape_info_stat[self.currentShape.shape] += 1
        return result

    #####################################
    ## 動かせるかどうか確認する
    # 動かせない場合 False を返す
    #####################################
    def tryMoveCurrent(self, direction, x, y):
        return self.tryMove(self.currentShape, direction, x, y)

    #####################################
    ## 初期位置へ2回配置して動かせなかったら Reset
    #####################################
    def tryMoveNext(self, direction, x, y):
        ret = self.tryMove(self.nextShape, direction, x, y)
        if ret == False:
            # if tryMove returns False 2 times, do reset.
            self.tryMoveNextCnt += 1
            if self.tryMoveNextCnt >= 2:
                self.tryMoveNextCnt = 0
                ret = True
            else:
                ret = False
        return ret

    #####################################
    # direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置可能か判定する
    # 配置できない場合 False を返す
    ######################################
    def tryMove(self, shape, direction, x, y):
        # direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置した場合の座標配列を繰り返す
        for x, y in shape.getCoords(direction, x, y):
            # 画面ボード外なので false
            if x >= BoardData.width or x < 0 or y >= BoardData.height or y < 0:
                return False
            # その位置にすでに他のブロックがあるので false
            if self.backBoard[x + y * BoardData.width] > 0:
                return False
        return True

    #####################################
    ## テノリミノを1つ落とし消去ラインとテトリミノ落下数を返す
    #####################################
    def moveDown(self):
        # move piece, 1 block
        # and return the number of lines which is removed in this function.
        removedlines = 0
        moveDownlines = 0
        # 動かせるか確認
        if self.tryMoveCurrent(self.currentDirection, self.currentX, self.currentY + 1):
            self.currentY += 1
            moveDownlines += 1
        # 動かせなくなったら確定
        else:
            ##画面ボードに固着したテトリミノを書き込む
            self.mergePiece()
            ## 画面ボードの消去できるラインを探して消去し、画面ボードを更新、そして消した Line を返す
            removedlines = self.removeFullLines()
            ## 新しい予告テトリミノ配列作成
            self.createNewPiece()
            
        return removedlines, moveDownlines

    #####################################
    ## テトリミノを一番下まで落とし消去ラインとテトリミノ落下数を返す
    #####################################
    def dropDown(self):
        # drop piece, immediately
        # and return the number of lines which is removed in this function.
        dropdownlines = 0
        while self.tryMoveCurrent(self.currentDirection, self.currentX, self.currentY + 1):
            self.currentY += 1
            dropdownlines += 1

        ##画面ボードに固着したテトリミノを書き込む
        self.mergePiece()
        ## 画面ボードの消去できるラインを探して消去し、画面ボードを更新、そして消した Line を返す
        removedlines = self.removeFullLines()
        ## 新しい予告テトリミノ配列作成
        self.createNewPiece()
        return removedlines, dropdownlines

    #####################################
    ## 左へテトリミノを1つ動かす
    ## 失敗したら Falase を返す
    #####################################
    def moveLeft(self):
        if self.tryMoveCurrent(self.currentDirection, self.currentX - 1, self.currentY):
            self.currentX -= 1
        else:
            #print("failed to moveLeft..")
            return False
        return True

    #####################################
    ## 右へテトリミノを1つ動かす
    ## 失敗したら Falase を返す
    #####################################
    def moveRight(self):
        if self.tryMoveCurrent(self.currentDirection, self.currentX + 1, self.currentY):
            self.currentX += 1
        else:
            #print("failed to moveRight..")
            return False
        return True

    #####################################
    ## 右回転させる
    #####################################
    def rotateRight(self):
        if self.tryMoveCurrent((self.currentDirection + 1) % 4, self.currentX, self.currentY):
            self.currentDirection += 1
            self.currentDirection %= 4
        else:
            #print("failed to rotateRight..")
            return False
        return True

    #####################################
    ## 左回転させる
    #####################################
    def rotateLeft(self):
        if self.tryMoveCurrent((self.currentDirection - 1) % 4, self.currentX, self.currentY):
            self.currentDirection -= 1
            self.currentDirection %= 4
        else:
            #print("failed to rotateLeft..")
            return False
        return True

    #####################################
    ## ホールド入れ替え
    #####################################
    def exchangeholdShape(self):
        if self.holdShape == None:
            # if holdShape not exists, set holdShape
            self.holdShape = self.currentShape
            self.createNewPiece()
            return False
        else:
            # if holdShape exists, exchange shapes
            self.holdShape,self.currentShape = self.currentShape,self.holdShape
            # init current X,Y,Direction
            minX, maxX, minY, maxY = self.nextShape.getBoundingOffsets(0)
            self.currentX = 5
            self.currentY = -minY
            self.currentDirection = 0
        return True

    #####################################
    ## 画面ボードの消去できるラインを探して消去し、画面ボードを更新、そして消した Line を返す
    #####################################
    def removeFullLines(self):
        newBackBoard = [0] * BoardData.width * BoardData.height
        newY = BoardData.height - 1
        # 消去ライン0
        lines = 0
        # 最下段行から探索
        for y in range(BoardData.height - 1, -1, -1):
            # y行のブロック数を数える
            blockCount = sum([1 if self.backBoard[x + y * BoardData.width] > 0 else 0 for x in range(BoardData.width)])
            # y行のブロック数が幅より少ない
            if blockCount < BoardData.width:
                # そのままコピー
                for x in range(BoardData.width):
                    newBackBoard[x + newY * BoardData.width] = self.backBoard[x + y * BoardData.width]
                newY -= 1
            # y行のブロックがうまっている
            else:
                # 消去ラインカウント+1
                lines += 1
        if lines > 0:
            self.backBoard = newBackBoard
        return lines


    #####################################
    ##画面ボードに固着したテトリミノを書き込む
    #####################################
    def mergePiece(self):
        # direction (回転状態)のテトリミノ座標配列を取得し、それをx,yに配置した場合の座標配列で繰り返す
        for x, y in self.currentShape.getCoords(self.currentDirection, self.currentX, self.currentY):
            # 画面ボードに書き込む
            self.backBoard[x + y * BoardData.width] = self.currentShape.shape

        # 現テトリミノ情報を初期化
        self.currentX = -1
        self.currentY = -1
        self.currentDirection = 0
        self.currentShape = Shape()

    #####################################
    ##画面ボードと現テトリミノ情報をクリア
    #####################################
    def clear(self):
        self.currentX = -1
        self.currentY = -1
        self.currentDirection = 0
        self.currentShape = Shape()
        self.backBoard = [0] * BoardData.width * BoardData.height
        self.addobstacle()

    #####################################
    ## 障害物配置
    #####################################
    def addobstacle(self):
        obstacle_height = self.obstacle_height
        obstacle_probability = self.obstacle_probability

        for y in range(BoardData.height):
            for x in range(BoardData.width):

                if y < (BoardData.height - obstacle_height):
                    continue

                # create obstacle
                tmp_num = np_randomObstacle.random.randint(1, 100)
                if tmp_num <= obstacle_probability:
                    self.backBoard[x + y * BoardData.width] = np_randomObstaclePiece.random.randint(1, 8)

BOARD_DATA = BoardData()
