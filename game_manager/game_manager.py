#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QMainWindow, QFrame, QDesktopWidget, QApplication, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor

from board_manager import BOARD_DATA, Shape
from block_controller import BLOCK_CONTROLLER
from block_controller_sample import BLOCK_CONTROLLER_SAMPLE

from argparse import ArgumentParser
import time
import json
import pprint

################################
# Option 取得
###############################
def get_option(game_time, mode, drop_interval, random_seed, obstacle_height, obstacle_probability, resultlogjson, train_yaml, predict_weight, user_name, ShapeListMax, BlockNumMax, art_config_filepath):
    argparser = ArgumentParser()
    argparser.add_argument('--game_time', type=int,
                           default=game_time,
                           help='Specify game time(s)')
    argparser.add_argument('--mode', type=str,
                           default=mode,
                           help='Specify mode (keyboard/gamepad/sample/train/art) if necessary')
    argparser.add_argument('--drop_interval', type=int,
                           default=drop_interval,
                           help='Specify drop_interval(s)')
    argparser.add_argument('--seed', type=int,
                           default=random_seed,
                           help='Specify random seed')
    argparser.add_argument('--obstacle_height', type=int,
                           default=obstacle_height,
                           help='Specify obstacle height')
    argparser.add_argument('--obstacle_probability', type=int,
                           default=obstacle_probability,
                           help='Specify obstacle probability')
    argparser.add_argument('--resultlogjson', type=str,
                           default=resultlogjson,
                           help='result json log file path')
    argparser.add_argument('--train_yaml', type=str,
                           default=train_yaml,
                           help='yaml file for machine learning')
    argparser.add_argument('--predict_weight', type=str,
                           default=predict_weight,
                           help='weight file for machine learning')
    argparser.add_argument('-u', '--user_name', type=str,
                           default=user_name,
                           help='Specigy user name if necessary')
    argparser.add_argument('--ShapeListMax', type=int,
                           default=ShapeListMax,
                           help='Specigy NextShapeNumberMax if necessary')
    argparser.add_argument('--BlockNumMax', type=int,
                           default=BlockNumMax,
                           help='Specigy BlockNumMax if necessary')
    argparser.add_argument('--art_config_filepath', type=str,
                           default=art_config_filepath,
                           help='art_config file path')

    return argparser.parse_args()

#####################################################################
#####################################################################
# Game Manager
#####################################################################
#####################################################################
class Game_Manager(QMainWindow):

    # a[n] = n^2 - n + 1
    LINE_SCORE_1 = 100
    LINE_SCORE_2 = 300
    LINE_SCORE_3 = 700
    LINE_SCORE_4 = 1300
    GAMEOVER_SCORE = -500

    ###############################################
    # 初期化
    ###############################################
    def __init__(self):
        super().__init__()
        self.isStarted = False
        self.isPaused = False
        self.nextMove = None
        self.lastShape = Shape.shapeNone

        self.game_time = -1
        self.block_index = 0
        self.mode = "default"
        self.drop_interval = 1000
        self.random_seed = time.time() * 10000000 # 0
        self.obstacle_height = 0
        self.obstacle_probability = 0
        self.ShapeListMax = 6
        self.BlockNumMax = -1
        self.resultlogjson = ""
        self.user_name = ""
        self.train_yaml = None
        self.predict_weight = None
        self.art_config_filepath = None
        
        args = get_option(self.game_time,
                          self.mode,
                          self.drop_interval,
                          self.random_seed,
                          self.obstacle_height,
                          self.obstacle_probability,
                          self.resultlogjson,
                          self.train_yaml,
                          self.predict_weight,
                          self.user_name,
                          self.ShapeListMax,
                          self.BlockNumMax,
                          self.art_config_filepath)
        if args.game_time >= 0:
            self.game_time = args.game_time
        if args.mode in ("keyboard", "gamepad", "sample", "art", "train", "predict", "train_sample", "predict_sample", "train_sample2", "predict_sample2"):
            self.mode = args.mode
        if args.drop_interval >= 0:
            self.drop_interval = args.drop_interval
        if args.seed >= 0:
            self.random_seed = args.seed
        if args.obstacle_height >= 0:
            self.obstacle_height = args.obstacle_height
        if args.obstacle_probability >= 0:
            self.obstacle_probability = args.obstacle_probability
        if len(args.resultlogjson) != 0:
            self.resultlogjson = args.resultlogjson
        if len(args.user_name) != 0:
            self.user_name = args.user_name
        if args.ShapeListMax > 0:
            self.ShapeListMax = args.ShapeListMax
        
        if args.BlockNumMax > 0:
            self.BlockNumMax = args.BlockNumMax
        if args.train_yaml.endswith('.yaml'):

            self.train_yaml = args.train_yaml        
        if args.predict_weight != "default":
            self.predict_weight = args.predict_weight
        if args.art_config_filepath.endswith('.json'):
            self.art_config_filepath = args.art_config_filepath      
            
        self.initUI()
        
    ###############################################
    # UI 初期化
    ###############################################
    def initUI(self):
        self.gridSize = 22
        self.NextShapeYOffset = 90
        # display maximum 4 next blocks
        self.NextShapeMaxAppear = min(4, self.ShapeListMax - 1)

        self.speed = self.drop_interval # block drop speed

        self.timer = QBasicTimer()
        self.setFocusPolicy(Qt.StrongFocus)

        hLayout = QHBoxLayout()

        random_seed_Nextshape = self.random_seed
        self.tboard = Board(self, self.gridSize,
                            self.game_time,
                            random_seed_Nextshape,
                            self.obstacle_height,
                            self.obstacle_probability,
                            self.ShapeListMax,
                            self.art_config_filepath)
        hLayout.addWidget(self.tboard)

        self.sidePanel = SidePanel(self, self.gridSize, self.NextShapeYOffset, self.NextShapeMaxAppear)
        hLayout.addWidget(self.sidePanel)

        self.statusbar = self.statusBar()
        self.tboard.msg2Statusbar[str].connect(self.statusbar.showMessage)

        self.start()

        self.center()

        WindowTitle = "Tetris"
        if len(self.user_name) != 0:
            WindowTitle = "Tetris_" + self.user_name
        self.setWindowTitle(WindowTitle)
        self.show()

        self.setFixedSize(self.tboard.width() + self.sidePanel.width(),
                          self.sidePanel.height() + self.statusbar.height())

    ###############################################
    # Window を中心へ移動
    ###############################################
    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    ###############################################
    # 開始
    ###############################################
    def start(self):
        if self.isPaused:
            return

        self.isStarted = True
        self.tboard.score = 0
        ##画面ボードと現テトリミノ情報をクリア
        BOARD_DATA.clear()
        ## 新しい予告テトリミノ配列作成
        BOARD_DATA.createNewPiece()
        self.tboard.msg2Statusbar.emit(str(self.tboard.score))
        self.timer.start(self.speed, self)

    ###############################################
    # ポーズ
    ###############################################
    def pause(self):
        if not self.isStarted:
            return

        self.isPaused = not self.isPaused

        if self.isPaused:
            self.timer.stop()
            self.tboard.msg2Statusbar.emit("paused")
        else:
            self.timer.start(self.speed, self)

        self.updateWindow()

    ###############################################
    # ゲームリセット (ゲームオーバー)
    ###############################################
    def resetfield(self):
        # self.tboard.score = 0
        self.tboard.reset_cnt += 1
        self.tboard.score += Game_Manager.GAMEOVER_SCORE
        ##画面ボードと現テトリミノ情報をクリア
        BOARD_DATA.clear()
        ## 新しい予告テトリミノ配列作成
        BOARD_DATA.createNewPiece()
        

    ###############################################
    # 画面リセット
    ###############################################
    def reset_all_field(self):
        # reset all field for debug
        # this function is mainly for machine learning
        self.tboard.reset_cnt = 0
        self.tboard.score = 0
        self.tboard.dropdownscore = 0
        self.tboard.linescore = 0
        self.tboard.line = 0
        self.tboard.line_score_stat = [0, 0, 0, 0]
        self.tboard.start_time = time.time()
        ##画面ボードと現テトリミノ情報をクリア
        BOARD_DATA.clear()
        ## 新しい予告テトリミノ配列作成
        BOARD_DATA.createNewPiece()

    ###############################################
    # Window 情報 UPDATE
    ###############################################
    def updateWindow(self):
        self.tboard.updateData()
        self.sidePanel.updateData()
        self.update()

    ###############################################
    # タイマーイベント
    ###############################################
    def timerEvent(self, event):
        # callback function for user control

        if event.timerId() == self.timer.timerId():
            next_x = 0
            next_y_moveblocknum = 0
            y_operation = -1

            if BLOCK_CONTROLLER and not self.nextMove:
                # update CurrentBlockIndex
                if BOARD_DATA.currentY <= 1:
                    self.block_index = self.block_index + 1

                # nextMove data structure
                nextMove = {"strategy":
                                {
                                  "direction": "none",    # next shape direction ( 0 - 3 )
                                  "x": "none",            # next x position (range: 0 - (witdh-1) )
                                  "y_operation": "none",  # movedown or dropdown (0:movedown, 1:dropdown)
                                  "y_moveblocknum": "none", # amount of next y movement
                                  "use_hold_function": "n", # use hold function (y:yes, n:no)
                                },
                            "option":
                                { "reset_callback_function_addr":None,
                                  "reset_all_field": None,
                                  "force_reset_field": None,
                                }
                            }
                # get nextMove from GameController
                GameStatus = self.getGameStatus()

                if self.mode == "sample":
                    # sample
                    self.nextMove = BLOCK_CONTROLLER_SAMPLE.GetNextMove(nextMove, GameStatus)

                elif self.mode == "train_sample" or self.mode == "predict_sample":
                    # sample train/predict
                    # import block_controller_train_sample, it's necessary to install pytorch to use.
                    from machine_learning.block_controller_train_sample import BLOCK_CONTROLLER_TRAIN_SAMPLE as BLOCK_CONTROLLER_TRAIN
                    self.nextMove = BLOCK_CONTROLLER_TRAIN.GetNextMove(nextMove, GameStatus,yaml_file=self.train_yaml,weight=self.predict_weight)
                    
                elif self.mode == "train_sample2" or self.mode == "predict_sample2":
                    # sample train/predict
                    # import block_controller_train_sample, it's necessary to install pytorch to use.
                    from machine_learning.block_controller_train_sample2 import BLOCK_CONTROLLER_TRAIN_SAMPLE2 as BLOCK_CONTROLLER_TRAIN
                    self.nextMove = BLOCK_CONTROLLER_TRAIN.GetNextMove(nextMove, GameStatus,yaml_file="config/train_sample2.yaml",weight=self.predict_weight)
                    
                elif self.mode == "train" or self.mode == "predict":
                    # train/predict
                    # import block_controller_train, it's necessary to install pytorch to use.
                    from machine_learning.block_controller_train import BLOCK_CONTROLLER_TRAIN
                    self.nextMove = BLOCK_CONTROLLER_TRAIN.GetNextMove(nextMove, GameStatus,yaml_file=self.train_yaml,weight=self.predict_weight)
                elif self.mode == "art":
                    # art
                    # print GameStatus
                    import pprint
                    print("=================================================>")
                    pprint.pprint(GameStatus, width = 61, compact = True)
                    # get direction/x/y from art_config
                    d,x,y = BOARD_DATA.getnextShapeIndexListDXY(self.block_index-1)
                    nextMove["strategy"]["direction"] = d
                    nextMove["strategy"]["x"] = x
                    nextMove["strategy"]["y_operation"] = y
                    nextMove["strategy"]["y_moveblocknum"] = 1
                    self.nextMove = nextMove
                else:
                    self.nextMove = BLOCK_CONTROLLER.GetNextMove(nextMove, GameStatus)

                if self.mode in ("keyboard", "gamepad"):
                    # ignore nextMove, for keyboard/gamepad controll
                    self.nextMove["strategy"]["x"] = BOARD_DATA.currentX
                    # Move Down 数
                    self.nextMove["strategy"]["y_moveblocknum"] = 1
                    # Drop Down:1, Move Down:0
                    self.nextMove["strategy"]["y_operation"] = 0
                    # テトリミノ回転数
                    self.nextMove["strategy"]["direction"] = BOARD_DATA.currentDirection

            #######################
            ## 次の手を動かす
            if self.nextMove:
                # shape direction operation
                next_x = self.nextMove["strategy"]["x"]
                # Move Down 数
                next_y_moveblocknum = self.nextMove["strategy"]["y_moveblocknum"]
                # Drop Down:1, Move Down:0
                y_operation = self.nextMove["strategy"]["y_operation"]
                # テトリミノ回転数
                next_direction = self.nextMove["strategy"]["direction"]
                use_hold_function = self.nextMove["strategy"]["use_hold_function"]

                # if use_hold_function
                if use_hold_function == "y":
                    isExchangeHoldShape = BOARD_DATA.exchangeholdShape()
                    if isExchangeHoldShape == False:
                        # if isExchangeHoldShape is False, this means no holdshape exists. 
                        # so it needs to return immediately to use new shape.
                        # init nextMove
                        self.nextMove = None
                        return

                k = 0
                while BOARD_DATA.currentDirection != next_direction and k < 4:
                    ret = BOARD_DATA.rotateRight()
                    if ret == False:
                        #print("cannot rotateRight")
                        break
                    k += 1
                # x operation
                k = 0
                while BOARD_DATA.currentX != next_x and k < 5:
                    if BOARD_DATA.currentX > next_x:
                        ret = BOARD_DATA.moveLeft()
                        if ret == False:
                            #print("cannot moveLeft")
                            break
                    elif BOARD_DATA.currentX < next_x:
                        ret = BOARD_DATA.moveRight()
                        if ret == False:
                            #print("cannot moveRight")
                            break
                    k += 1

            # dropdown/movedown lines
            dropdownlines = 0
            removedlines = 0
            if y_operation == 1: # dropdown
                ## テトリミノを一番下まで落とす
                removedlines, dropdownlines = BOARD_DATA.dropDown()
            else: # movedown, with next_y_moveblocknum lines
                k = 0
                # Move down を1つずつ処理
                while True:
                    ## テノリミノを1つ落とし消去ラインとテトリミノ落下数を返す
                    removedlines, movedownlines = BOARD_DATA.moveDown()
                    # Drop してたら除外 (テトリミノが1つも落下していない場合)
                    if movedownlines < 1:
                        # if already dropped
                        break
                    k += 1
                    if k >= next_y_moveblocknum:
                        # if already movedown next_y_moveblocknum block
                        break

            # 消去ライン数と落下数によりスコア計算
            self.UpdateScore(removedlines, dropdownlines)

            ##############################
            #
            # check reset field
            #if BOARD_DATA.currentY < 1: 
            if BOARD_DATA.currentY < 1 or self.nextMove["option"]["force_reset_field"] == True:
                # if Piece cannot movedown and stack, reset field
                if self.nextMove["option"]["reset_callback_function_addr"] != None:
                    # if necessary, call reset_callback_function
                    reset_callback_function = self.nextMove["option"]["reset_callback_function_addr"]
                    reset_callback_function()

                if self.nextMove["option"]["reset_all_field"] == True:
                    # reset all field if debug option is enabled
                    print("reset all field.")
                    self.reset_all_field()
                else:
                    # ゲームリセット = ゲームオーバー
                    self.resetfield()

            # init nextMove
            self.nextMove = None

            # update window
            self.updateWindow()
            return
        else:
            super(Game_Manager, self).timerEvent(event)

    ###############################################
    # 消去ライン数と落下数によりスコア計算
    ###############################################
    def UpdateScore(self, removedlines, dropdownlines):
        # calculate and update current score
        # 消去ライン数で計算
        if removedlines == 1:
            linescore = Game_Manager.LINE_SCORE_1
        elif removedlines == 2:
            linescore = Game_Manager.LINE_SCORE_2
        elif removedlines == 3:
            linescore = Game_Manager.LINE_SCORE_3
        elif removedlines == 4:
            linescore = Game_Manager.LINE_SCORE_4
        else:
            linescore = 0
        # 落下スコア計算
        dropdownscore = dropdownlines
        self.tboard.dropdownscore += dropdownscore
        # 合計計算
        self.tboard.linescore += linescore
        self.tboard.score += ( linescore + dropdownscore )
        self.tboard.line += removedlines
        # 同時消去数をカウント
        if removedlines > 0:
            self.tboard.line_score_stat[removedlines - 1] += 1

    ###############################################
    # ゲーム情報の取得
    ###############################################
    def getGameStatus(self):
        # return current Board status.
        # define status data.
        status = {"field_info":
                      {
                        "width": "none",
                        "height": "none",
                        "backboard": "none",
                        "withblock": "none", # back board with current block
                      },
                  "block_info":
                      {
                        "currentX":"none",
                        "currentY":"none",
                        "currentDirection":"none",
                        "currentShape":{
                           "class":"none",
                           "index":"none",
                           "direction_range":"none",
                        },
                        "nextShape":{
                           "class":"none",
                           "index":"none",
                           "direction_range":"none",
                        },
                        "nextShapeList":{
                        },
                        "holdShape":{
                           "class":"none",
                           "index":"none",
                           "direction_range":"none",
                        },
                      },
                  "judge_info":
                      {
                        "elapsed_time":"none",
                        "game_time":"none",
                        "gameover_count":"none",
                        "score":"none",
                        "line":"none",
                        "block_index":"none",
                        "block_num_max":"none",
                        "mode":"none",
                      },
                  "debug_info":
                      {
                        "dropdownscore":"none",
                        "linescore":"none",
                        "line_score": {
                          "line1":"none",
                          "line2":"none",
                          "line3":"none",
                          "line4":"none",
                          "gameover":"none",
                        },
                        "shape_info": {
                          "shapeNone": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeI": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeL": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeJ": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeT": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeO": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeS": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeZ": {
                             "index" : "none",
                             "color" : "none",
                          },
                        },
                        "line_score_stat":"none",
                        "shape_info_stat":"none",
                        "random_seed":"none",
                        "obstacle_height":"none",
                        "obstacle_probability":"none"
                      },
                  }
        # update status
        ## board
        status["field_info"]["width"] = BOARD_DATA.width
        status["field_info"]["height"] = BOARD_DATA.height
        status["field_info"]["backboard"] = BOARD_DATA.getData()
        status["field_info"]["withblock"] = BOARD_DATA.getDataWithCurrentBlock()
        ## shape
        status["block_info"]["currentX"] = BOARD_DATA.currentX
        status["block_info"]["currentY"] = BOARD_DATA.currentY
        status["block_info"]["currentDirection"] = BOARD_DATA.currentDirection
        ### current shape
        currentShapeClass, currentShapeIdx, currentShapeRange = BOARD_DATA.getShapeData(0)
        status["block_info"]["currentShape"]["class"] = currentShapeClass
        status["block_info"]["currentShape"]["index"] = currentShapeIdx
        status["block_info"]["currentShape"]["direction_range"] = currentShapeRange
        ### next shape
        nextShapeClass, nextShapeIdx, nextShapeRange = BOARD_DATA.getShapeData(1)
        status["block_info"]["nextShape"]["class"] = nextShapeClass
        status["block_info"]["nextShape"]["index"] = nextShapeIdx
        status["block_info"]["nextShape"]["direction_range"] = nextShapeRange
        ### next shape list
        for i in range(BOARD_DATA.getShapeListLength()):
            ElementNo="element" + str(i)
            ShapeClass, ShapeIdx, ShapeRange = BOARD_DATA.getShapeData(i)
            status["block_info"]["nextShapeList"][ElementNo] = {
                "class":ShapeClass,
                "index":ShapeIdx,
                "direction_range":ShapeRange,
            }
        ### hold shape
        holdShapeClass, holdShapeIdx, holdShapeRange = BOARD_DATA.getholdShapeData()
        status["block_info"]["holdShape"]["class"] = holdShapeClass
        status["block_info"]["holdShape"]["index"] = holdShapeIdx
        status["block_info"]["holdShape"]["direction_range"] = holdShapeRange
        ### next shape
        ## judge_info
        status["judge_info"]["elapsed_time"] = round(time.time() - self.tboard.start_time, 3)
        status["judge_info"]["game_time"] = self.game_time
        status["judge_info"]["gameover_count"] = self.tboard.reset_cnt
        status["judge_info"]["score"] = self.tboard.score
        status["judge_info"]["line"] = self.tboard.line
        status["judge_info"]["block_index"] = self.block_index
        status["judge_info"]["block_num_max"] = self.BlockNumMax
        status["judge_info"]["mode"] = self.mode
        ## debug_info
        status["debug_info"]["dropdownscore"] = self.tboard.dropdownscore
        status["debug_info"]["linescore"] = self.tboard.linescore
        status["debug_info"]["line_score_stat"] = self.tboard.line_score_stat
        status["debug_info"]["shape_info_stat"] = BOARD_DATA.shape_info_stat
        status["debug_info"]["line_score"]["line1"] = Game_Manager.LINE_SCORE_1
        status["debug_info"]["line_score"]["line2"] = Game_Manager.LINE_SCORE_2
        status["debug_info"]["line_score"]["line3"] = Game_Manager.LINE_SCORE_3
        status["debug_info"]["line_score"]["line4"] = Game_Manager.LINE_SCORE_4
        status["debug_info"]["line_score"]["gameover"] = Game_Manager.GAMEOVER_SCORE
        status["debug_info"]["shape_info"]["shapeNone"]["index"] = Shape.shapeNone
        status["debug_info"]["shape_info"]["shapeI"]["index"] = Shape.shapeI
        status["debug_info"]["shape_info"]["shapeI"]["color"] = "red"
        status["debug_info"]["shape_info"]["shapeL"]["index"] = Shape.shapeL
        status["debug_info"]["shape_info"]["shapeL"]["color"] = "green"
        status["debug_info"]["shape_info"]["shapeJ"]["index"] = Shape.shapeJ
        status["debug_info"]["shape_info"]["shapeJ"]["color"] = "purple"
        status["debug_info"]["shape_info"]["shapeT"]["index"] = Shape.shapeT
        status["debug_info"]["shape_info"]["shapeT"]["color"] = "gold"
        status["debug_info"]["shape_info"]["shapeO"]["index"] = Shape.shapeO
        status["debug_info"]["shape_info"]["shapeO"]["color"] = "pink"
        status["debug_info"]["shape_info"]["shapeS"]["index"] = Shape.shapeS
        status["debug_info"]["shape_info"]["shapeS"]["color"] = "blue"
        status["debug_info"]["shape_info"]["shapeZ"]["index"] = Shape.shapeZ
        status["debug_info"]["shape_info"]["shapeZ"]["color"] = "yellow"
        status["debug_info"]["random_seed"] = self.random_seed
        status["debug_info"]["obstacle_height"] = self.obstacle_height
        status["debug_info"]["obstacle_probability"] = self.obstacle_probability
        if currentShapeIdx == Shape.shapeNone:
            print("warning: current shape is none !!!")

        return status

    def getGameStatusJson(self):
        status = {
                  "debug_info":
                      {
                        "line_score": {
                          "line1":"none",
                          "line2":"none",
                          "line3":"none",
                          "line4":"none",
                          "gameover":"none",
                        },
                        "shape_info": {
                          "shapeNone": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeI": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeL": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeJ": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeT": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeO": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeS": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeZ": {
                             "index" : "none",
                             "color" : "none",
                          },
                        },
                        "line_score_stat":"none",
                        "shape_info_stat":"none",
                        "random_seed":"none",
                        "obstacle_height":"none",
                        "obstacle_probability":"none",
                      },
                  "judge_info":
                      {
                        "elapsed_time":"none",
                        "game_time":"none",
                        "gameover_count":"none",
                        "score":"none",
                        "line":"none",
                        "block_index":"none",
                        "block_num_max":"none",
                        "mode":"none",
                      },
                  }
        # update status
        ## debug_info
        status["debug_info"]["line_score_stat"] = self.tboard.line_score_stat
        status["debug_info"]["shape_info_stat"] = BOARD_DATA.shape_info_stat
        status["debug_info"]["line_score"]["line1"] = Game_Manager.LINE_SCORE_1
        status["debug_info"]["line_score"]["line2"] = Game_Manager.LINE_SCORE_2
        status["debug_info"]["line_score"]["line3"] = Game_Manager.LINE_SCORE_3
        status["debug_info"]["line_score"]["line4"] = Game_Manager.LINE_SCORE_4
        status["debug_info"]["line_score"]["gameover"] = Game_Manager.GAMEOVER_SCORE
        status["debug_info"]["shape_info"]["shapeNone"]["index"] = Shape.shapeNone
        status["debug_info"]["shape_info"]["shapeI"]["index"] = Shape.shapeI
        status["debug_info"]["shape_info"]["shapeI"]["color"] = "red"
        status["debug_info"]["shape_info"]["shapeL"]["index"] = Shape.shapeL
        status["debug_info"]["shape_info"]["shapeL"]["color"] = "green"
        status["debug_info"]["shape_info"]["shapeJ"]["index"] = Shape.shapeJ
        status["debug_info"]["shape_info"]["shapeJ"]["color"] = "purple"
        status["debug_info"]["shape_info"]["shapeT"]["index"] = Shape.shapeT
        status["debug_info"]["shape_info"]["shapeT"]["color"] = "gold"
        status["debug_info"]["shape_info"]["shapeO"]["index"] = Shape.shapeO
        status["debug_info"]["shape_info"]["shapeO"]["color"] = "pink"
        status["debug_info"]["shape_info"]["shapeS"]["index"] = Shape.shapeS
        status["debug_info"]["shape_info"]["shapeS"]["color"] = "blue"
        status["debug_info"]["shape_info"]["shapeZ"]["index"] = Shape.shapeZ
        status["debug_info"]["shape_info"]["shapeZ"]["color"] = "yellow"
        status["debug_info"]["random_seed"] = self.random_seed
        status["debug_info"]["obstacle_height"] = self.obstacle_height
        status["debug_info"]["obstacle_probability"] = self.obstacle_probability
        ## judge_info
        status["judge_info"]["elapsed_time"] = round(time.time() - self.tboard.start_time, 3)
        status["judge_info"]["game_time"] = self.game_time
        status["judge_info"]["gameover_count"] = self.tboard.reset_cnt
        status["judge_info"]["score"] = self.tboard.score
        status["judge_info"]["line"] = self.tboard.line
        status["judge_info"]["block_index"] = self.block_index
        status["judge_info"]["block_num_max"] = self.BlockNumMax
        status["judge_info"]["mode"] = self.mode
        return json.dumps(status)

    ###############################################
    # キー入力イベント処理 -m keyboard, gamepad
    # QMainWindow 継承
    ###############################################
    def keyPressEvent(self, event):
        # for keyboard/gamepad control

        # スタート前はキーキャプチャしない
        if not self.isStarted or BOARD_DATA.currentShape == Shape.shapeNone:
            super(Game_Manager, self).keyPressEvent(event)
            return

        key = event.key()
        
        # key event handle process.
        # depends on self.mode, it's better to make key config file.
        #  "keyboard" : PC keyboard controller
        #  "gamepad" : game controller. KeyUp, space are different

        if key == Qt.Key_P:
            self.pause()
            return
            
        if self.isPaused:
            return
        elif key == Qt.Key_Left:
            BOARD_DATA.moveLeft()
        elif key == Qt.Key_Right:
            BOARD_DATA.moveRight()
        elif (key == Qt.Key_Up and self.mode == 'keyboard') or (key == Qt.Key_Space and self.mode == 'gamepad'):
            BOARD_DATA.rotateLeft()
        elif key == Qt.Key_M:
            ## テノリミノを1つ落とし消去ラインとテトリミノ落下数を返す
            removedlines, movedownlines = BOARD_DATA.moveDown()
            # 消去ライン数によりスコア計算
            self.UpdateScore(removedlines, 0)
        elif (key == Qt.Key_Space and self.mode == 'keyboard') or (key == Qt.Key_Up and self.mode == 'gamepad'):
            ## テトリミノを一番下まで落とす
            removedlines, dropdownlines = BOARD_DATA.dropDown()
            # 消去ライン数と落下数によりスコア計算
            self.UpdateScore(removedlines, dropdownlines)
        elif key == Qt.Key_C:
            BOARD_DATA.exchangeholdShape()
        else:
            # スタート前はキーキャプチャしない
            super(Game_Manager, self).keyPressEvent(event)

        self.updateWindow()


###############################################
# 四角形の描画
###############################################
def drawSquare(painter, x, y, val, s):
    colorTable = BOARD_DATA.getcolorTable()

    # treat values as integer explicitly
    x = int(x)
    y = int(y)
    val = int(val)
    s = int(s)

    if val == 0:
        return

    color = QColor(colorTable[val])
    painter.fillRect(x + 1, y + 1, s - 2, s - 2, color)

    painter.setPen(color.lighter())
    painter.drawLine(x, y + s - 1, x, y)
    painter.drawLine(x, y, x + s - 1, y)

    painter.setPen(color.darker())
    painter.drawLine(x + 1, y + s - 1, x + s - 1, y + s - 1)
    painter.drawLine(x + s - 1, y + s - 1, x + s - 1, y + 1)


###############################################
###############################################
# 横画面描画
###############################################
###############################################
class SidePanel(QFrame):
    ###############################################
    # 初期化
    ###############################################
    def __init__(self, parent, gridSize, NextShapeYOffset, NextShapeMaxAppear):
        super().__init__(parent)
        self.setFixedSize(gridSize * 5, gridSize * BOARD_DATA.height)
        self.move(gridSize * BOARD_DATA.width, 0)
        self.gridSize = gridSize
        self.NextShapeYOffset = NextShapeYOffset
        self.NextShapeMaxAppear = NextShapeMaxAppear

    ###############################################
    # UPDATE
    ###############################################
    def updateData(self):
        self.update()

    ###############################################
    # 描画イベント
    ###############################################
    def paintEvent(self, event):
        painter = QPainter(self)

        ShapeListLength = BOARD_DATA.getShapeListLength()
        
        # draw next shape
        for i in range(ShapeListLength):
            if i == 0:
                # skip current shape
                continue
            if i > self.NextShapeMaxAppear:
                break

            ShapeClass, ShapeIdx, ShapeRange = BOARD_DATA.getShapeData(i) # nextShape

            # テトリミノが原点から x,y 両方向に最大何マス占有するのか取得
            minX, maxX, minY, maxY = ShapeClass.getBoundingOffsets(0)

            dy = 1 * self.gridSize
            dx = (self.width() - (maxX - minX) * self.gridSize) / 2
            
            val = ShapeClass.shape
            y_offset = self.NextShapeYOffset * (i - 1) #(self.NextShapeMaxAppear - i)
            # テトリミノを配置すべき座標リストを取得していく
            for x, y in ShapeClass.getCoords(0, 0, -minY):
                drawSquare(painter, x * self.gridSize + dx, y * self.gridSize + dy + y_offset, val, self.gridSize)

        # draw hold block area
        painter.setPen(QColor(0x777777))
        height_offset = self.height() - int(self.gridSize*4.65)
        painter.drawLine(0, height_offset,
                         self.width(), height_offset)
        painter.drawText(0, self.height(), 'HOLD');
        holdShapeClass, holdShapeIdx, holdShapeRange = BOARD_DATA.getholdShapeData()
        if holdShapeClass != None:
            # if holdShape exists, try to draw
            minX, maxX, minY, maxY = holdShapeClass.getBoundingOffsets(0)
            dy = 1 * self.gridSize
            dx = (self.width() - (maxX - minX) * self.gridSize) / 2
            val = holdShapeClass.shape
            y_offset = self.NextShapeYOffset * 4
            for x, y in holdShapeClass.getCoords(0, 0, -minY):
                drawSquare(painter, x * self.gridSize + dx, y * self.gridSize + dy + y_offset, val, self.gridSize)

#####################################################################
#####################################################################
# 画面ボード描画
#####################################################################
#####################################################################
class Board(QFrame):
    msg2Statusbar = pyqtSignal(str)

    ###############################################
    # 初期化
    ###############################################
    def __init__(self, parent, gridSize, game_time, random_seed, obstacle_height, obstacle_probability, ShapeListMax, art_config_filepath):
        super().__init__(parent)
        self.setFixedSize(gridSize * BOARD_DATA.width, gridSize * BOARD_DATA.height)
        self.gridSize = gridSize
        self.game_time = game_time
        self.initBoard(random_seed, obstacle_height, obstacle_probability, ShapeListMax, art_config_filepath)

    ###############################################
    # 画面ボード初期化
    ###############################################
    def initBoard(self, random_seed_Nextshape, obstacle_height, obstacle_probability, ShapeListMax, art_config_filepath):
        self.score = 0
        self.dropdownscore = 0
        self.linescore = 0
        self.line = 0
        self.line_score_stat = [0, 0, 0, 0]
        self.reset_cnt = 0
        self.start_time = time.time() 
        ##画面ボードと現テトリミノ情報をクリア
        BOARD_DATA.clear()
        BOARD_DATA.init_randomseed(random_seed_Nextshape)
        BOARD_DATA.init_obstacle_parameter(obstacle_height, obstacle_probability)
        BOARD_DATA.init_shape_parameter(ShapeListMax)
        BOARD_DATA.init_art_config(art_config_filepath)

    ###############################################
    # 描画イベント
    ###############################################
    def paintEvent(self, event):
        painter = QPainter(self)

        # Draw backboard
        for x in range(BOARD_DATA.width):
            for y in range(BOARD_DATA.height):
                val = BOARD_DATA.getValue(x, y)
                drawSquare(painter, x * self.gridSize, y * self.gridSize, val, self.gridSize)

        # Draw current shape
        for x, y in BOARD_DATA.getCurrentShapeCoord():
            val = BOARD_DATA.currentShape.shape
            drawSquare(painter, x * self.gridSize, y * self.gridSize, val, self.gridSize)

        # Draw a border
        painter.setPen(QColor(0x777777))
        painter.drawLine(self.width()-1, 0, self.width()-1, self.height())
        painter.setPen(QColor(0xCCCCCC))
        painter.drawLine(self.width(), 0, self.width(), self.height())

    ###############################################
    # ログファイル出力
    ###############################################
    def OutputLogData(self, isPrintLog):
        log_file_path = GAME_MANEGER.resultlogjson
        if len(log_file_path) != 0:
            if isPrintLog:
                print("##### OUTPUT_RESULT_LOG_FILE #####")
                print(log_file_path)
            with open(log_file_path, "w") as f:
                GameStatusJson = GAME_MANEGER.getGameStatusJson()
                f.write(GameStatusJson)

    ###############################################
    # データ更新
    ###############################################
    def updateData(self):
        score_str = str(self.score)
        line_str = str(self.line)
        reset_cnt_str = str(self.reset_cnt)
        elapsed_time = round(time.time() - self.start_time, 3)
        elapsed_time_str = str(elapsed_time)
        status_str = "score:" + score_str + ",line:" + line_str + ",gameover:" + reset_cnt_str + ",time[s]:" + elapsed_time_str

        # get gamestatus info
        GameStatus = GAME_MANEGER.getGameStatus()
        current_block_index = GameStatus["judge_info"]["block_index"]
        BlockNumMax = GameStatus["judge_info"]["block_num_max"]

        # print string to status bar
        self.msg2Statusbar.emit(status_str)
        self.update()
        self.OutputLogData(isPrintLog = False)

        if self.game_time == -1:
            pass
            #print("game_time: {}".format(self.game_time))
            #print("endless loop")
        elif (self.game_time >= 0 and elapsed_time >= self.game_time) or (current_block_index == BlockNumMax):
            # finish game.
            # 1. if elapsed_time beyonds given game_time.
            # 2. if current_block_index beyonds given BlockNumMax.
            print("game finish!! elapsed time: " + elapsed_time_str + "/game_time: " + str(self.game_time) \
                  + ", " + "current_block_index: " + str(current_block_index) + "/BlockNumMax: " + str(BlockNumMax))
            print("")
            print("##### YOUR_RESULT #####")
            print(status_str)
            print("")
            print("##### SCORE DETAIL #####")
            GameStatus = GAME_MANEGER.getGameStatus()
            line_score_stat = GameStatus["debug_info"]["line_score_stat"]
            line_Score = GameStatus["debug_info"]["line_score"]
            gameover_count = GameStatus["judge_info"]["gameover_count"]
            score = GameStatus["judge_info"]["score"]
            dropdownscore = GameStatus["debug_info"]["dropdownscore"]
            print("  1 line: " + str(line_Score["line1"]) + " * " + str(line_score_stat[0]) + " = " + str(line_Score["line1"] * line_score_stat[0]))
            print("  2 line: " + str(line_Score["line2"]) + " * " + str(line_score_stat[1]) + " = " + str(line_Score["line2"] * line_score_stat[1]))
            print("  3 line: " + str(line_Score["line3"]) + " * " + str(line_score_stat[2]) + " = " + str(line_Score["line3"] * line_score_stat[2]))
            print("  4 line: " + str(line_Score["line4"]) + " * " + str(line_score_stat[3]) + " = " + str(line_Score["line4"] * line_score_stat[3]))
            print("  dropdownscore: " + str(dropdownscore))
            print("  gameover: : " + str(line_Score["gameover"]) + " * " + str(gameover_count) + " = " + str(line_Score["gameover"] * gameover_count))

            print("##### ###### #####")
            print("")
            self.OutputLogData(isPrintLog = True)

            #sys.exit(app.exec_())
            sys.exit(0)

if __name__ == '__main__':
    app = QApplication([])
    GAME_MANEGER = Game_Manager()
    sys.exit(app.exec_())
